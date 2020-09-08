function [HWeights, JWeights, EnergyGap, GStates, SuccessFlag] = PEPGSM(NumLabel,NumVertex,NumGS,HLim, JLim, LabelPairCost,LabelSingletCost,Connection)
%Function to estimate the Energy parameters in Pott's model given the
%number of ground states
%Input: 
%   NumLabel = Number of Labels
%   NumVertex = Number of Vertex 
%   NumGS = Number of Ground States
%   HLim = [min(H) max(H)]
%   JLim = [min(J) max(J)]
%Optional Input: 
%   LabelPairCost = A symmetric matrix of size NumLabel x NumLabel with
%                   (i,j) elements = V(S_i, S_j)
%   LabelSingletCost = An array of size NumLabel with i^th element = U(S_i)
%   Connection = An upper triangle matrix NumVertex x NumVertex with (i,j) 
%               elements = 0 in the absence of connection between i and j. 
%Output: 
%   HWeights = An array of size NumVertices x 1 representing Field
%              parameters
%   JWeights = An upper triangular sparse matrix with (i,j) elements as the
%              edge weight of (i,j) connection
%   EnergyGap = GroundStateEnergy - LowestExcitedStateEnergy
%   GStates = A matrix of size NumVertices x NumGroundStates  with each 
%               elementin the set {1,2,...,NumLabel}. Each column 
%               represents a ground state
%   SuccessFlag = 1 if Parameters were found, 0 otherwise
%-------------------------------------------------------------------------
% Setting up optional inputs
%-------------------------------------------------------------------------
if nargin<7
    % Fully connected graph
    Connection = triu(ones(NumVertex),1);
else
    assert(size(Connection,1)==NumVertex && size(Connection,2)==NumVertex,...
        'Connection matrix should be a square matrix of size NumV\n');
    Connection = triu(double(Connection~=0),1); 
end

if nargin<6
    % U(S_i) is linearly graded between -1 and 1
    LabelSingletCost = linspace(-1,1,NumLabel);
else
    assert(length(LabelSingletCost)==NumLabel,...
        'LabelSingletCost should be a vector of size NumLabel\n');
end

if nargin<5
    % V(S_i, S_j) = 1 for all i != j 
    LabelPairCost = ones(NumLabel) - eye(NumLabel);
else
    assert(size(LabelPairCost,1)==NumLabel && size(LabelPairCost,2)==NumLabel, ...
    'LabelPairCost should be a square matrix of size NLabel\n' )
    LabelPairCost = (LabelPairCost+LabelPairCost')/2; %Turn into symmetric matrix
end

%-------------------------------------------------------------------------
% Setting up useful variables
%-------------------------------------------------------------------------

%Matrices for indexing Edges
[NumEdge,EdgeMap,EdgeMapInverse] = IndexEdges(NumVertex,Connection);

% M = A large number to be used in MILP
M = 2.5*(max(abs(HLim))*NumVertex + max(abs(JLim))*max(max(LabelPairCost))*NumEdge);

NumTS = NumLabel^NumVertex;  %Number of Total States

assert(NumGS<=NumTS, 'Too many states ')

% SetIndex: Index of states
SetIndex = 1:NumTS;
States = Index2State(SetIndex, NumVertex, NumLabel);

%Number of parameters
NumParam = NumVertex + NumEdge;

% StateSet = NumParam x NumTS with kth column = [V(S_i,S_j);U(S_i)] where S is
%         the array of labels corresponding to kth state 

StateSet = zeros(NumParam,NumTS);


for i=1:NumVertex
    StateSet(i,:)=LabelSingletCost(States(i,SetIndex));
end
for k=1:NumEdge
    i = EdgeMap(k,1);
    j = EdgeMap(k,2);
    temp = (States(j,SetIndex)-1)*NumLabel + (States(i,SetIndex));
    StateSet(NumVertex+EdgeMapInverse(j,i),:)=LabelPairCost(temp);
end


Jlb = JLim(1);
Jub = JLim(2);
Hlb = HLim(1);
Hub = HLim(2);

%-------------------------------------------------------------------------
%Matrices for Mixed Integer Linear Programming
%-------------------------------------------------------------------------
% Variables: [ {J_ij}, {H_i}, E_1, E_0, {l_i}, {m_i}]
%             J_ij - Edge strength of (ij) connection
%             H_i - Field strength of i-vertex
%             E_0 - Ground state energy
%             E_1 - Energy of lowest Excited state
%             l_i - {0/1} variable corresponding Ground state
%                    l_i = 1 if ith index is a ground state        
%                    l_i = 0 if otherwise
%             m_i - {0/1} variable corresponding ith Excited state
%                    m_i = 1 if ith index is the lowest excited states            
%                    m_i = 0 if otherwise
display('Estimating matrices for MILP ')
% Objective function 
% Cost = E_0 - E_1
f = [zeros(NumParam,1); 1; -1; sparse(NumTS,1); sparse(NumTS,1)];

% Inequality conditions: 
% -E(kth excited state) + E_0                 <= 0 for all k
%  E(kth excited state) - E_0 + M*p_k         <= M for all k
% -E(kth excited state) + E_1 - M*p_k         <= 0 for all k
%  E(kth excited state) - E_1         + M*r_k <= M for all k
%                                 p_k +   r_k <= 1 for all k
A = sparse([[            -StateSet', ones(NumTS,1),zeros(NumTS,1),sparse(NumTS,NumTS),sparse(NumTS,NumTS)]; ...
            [             StateSet',-ones(NumTS,1),zeros(NumTS,1),     M*speye(NumTS),sparse(NumTS,NumTS)]; ...
            [            -StateSet',zeros(NumTS,1), ones(NumTS,1),    -M*speye(NumTS),sparse(NumTS,NumTS)]; ...
            [             StateSet',zeros(NumTS,1),-ones(NumTS,1),sparse(NumTS,NumTS),     M*speye(NumTS)]; ...
            [zeros(NumTS, NumParam),zeros(NumTS,1),zeros(NumTS,1),       speye(NumTS),       speye(NumTS)]]);
b = [   zeros(NumTS,1);...
       M*ones(NumTS,1);...
        zeros(NumTS,1);...
       M*ones(NumTS,1);...
         ones(NumTS,1);];

% Equality conditions
%  p_1 + p_2 + ... + p_NumTS = NumGS
%  q_1 + q_2 + ... + q_NumTS = 1
Aeq =sparse( [[zeros(1, NumParam),                 0,             0, ones(1,NumTS),zeros(1,NumTS)];...
              [zeros(1, NumParam),                 0,             0,zeros(1,NumTS), ones(1,NumTS)]]);
beq = [NumGS;...
           1;];

% Bounds
%  JLim(1)<= J_ij <= JLim(2)
%  HLim(1)<=  H_i <= HLim(2)
%      -M <=  E_1 <= M
%      -M <=  E_0 <= M
%       0 <=  l_i <= 1
%       0 <=  m_i <= 1
lb = [Hlb*ones(NumVertex,1);Jlb*ones(NumEdge,1);-M;-M;zeros(NumTS,1);zeros(NumTS,1)];
ub = [Hub*ones(NumVertex,1);Jub*ones(NumEdge,1); M; M; ones(NumTS,1); ones(NumTS,1)];

% Integera variables: {l_i}, {m_i}
intcon = NumParam + 2 + 1: NumParam + 2 + NumTS + NumTS ;

%-------------------------------------------------------------------------
% Linear Programming step
%-------------------------------------------------------------------------
display('Starting Linear programming step')
[x,fval] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub);

%-------------------------------------------------------------------------
% Post-Processing step
%-------------------------------------------------------------------------
HWeights = x(1: NumVertex);
JWeights = sparse(EdgeMap(:,1),EdgeMap(:,2),x(NumVertex + 1:...
    NumVertex+ NumEdge),NumVertex,NumVertex);
GStateIndex = find(x( NumParam + 2 + 1: NumParam + 2 + NumTS)>1-1e-4); 
GStates = Index2State(GStateIndex, NumVertex, NumLabel);

SuccessFlag = 1;
EnergyGap = -fval;
if fval>=0
    warning('No Minimizer found to replicate Ground state')    
    SuccessFlag = 0;
    EnergyGap = 0;
end

fprintf('Found Energy gap of %d\n',-fval);
display(GStates);
end

function [Index] = State2Index(State, NumV, NumLabel)
%Function to index each state from 1: NumLabel^NumV
% Change State notation (1 to NLabel) to Base notation (0 to NLabel-1)
State = State-1; 
Index = zeros(1,size(State,2)); 

for i = 1:NumV
    Index = Index + (NumLabel^(i-1))*(State(i,:));
end
Index = Index+1; 
end

function [State] = Index2State(Index, NumV, NumLabel)
%Function to estimate state from index
Index = Index(:).' ; %Change to vector
Index = Index-1; %Change values to : 0 to NumLabel^NumV - 1 
State = zeros(NumV,length(Index));
for i = 1:NumV
    State(i,:) = rem(Index,NumLabel);
    Index = (Index - State(i,:))/NumLabel;
end
State = State+1; 
end

function [NumEdge,EdgeMap,EdgeMapInverse] = IndexEdges(NumVertex,Connection)
%Function to create maps for indexing Edges
%Output:
%   NumEdge = Number of Edges
%   EdgeMap = A matrix of size NumEdge x 2 where kth row gives the vertex
%             of kth edge
%   EdgeMapInverse = A matrix of size NumVertex x Numvertex where (ij)
%                    element gives the index of the edge-(ij)
NumEdge = sum(sum(Connection));
EdgeMapInverse = Connection;
EdgeMap = zeros(NumEdge,2);
temp = find(Connection==1);
for i=1:length(temp)
    tempx = 1 + mod(temp(i)-1 ,NumVertex);
    tempy = 1+ ((temp(i)-tempx) / (NumVertex));
    EdgeMap(i,:) = [tempx, tempy];
    EdgeMapInverse(tempx, tempy) = i;
    EdgeMapInverse(tempy, tempx) = i;
end
end