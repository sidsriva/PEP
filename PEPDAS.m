function [HWeights, JWeights, EnergyGap,SuccessFlag] = PEPDAS(NumLabel,DataStates,HLim, JLim, LabelPairCost,LabelSingletCost,Connection)
%Function to estimate the Energy parameters in Pott's model given the
%data states
%Input:
%   NumLabel = Number of Labels
%   DataStates = A matrix of size NumVertices x NumGroundStates  with each
%               elementin the set {1,2,...,NumLabel}. Each column
%               represents a data state
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
%   SuccessFlag = 1 if Parameters were found, 0 otherwise
display(DataStates);
%-------------------------------------------------------------------------
% Setting up optional inputs
%-------------------------------------------------------------------------
NumVertex = size(DataStates,1);
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
[NumEdge,EdgeMap,~] = IndexEdges(NumVertex,Connection);

% M = A large number to be used in MILP
M = 2.5*(max(abs(HLim))*NumVertex + max(abs(JLim))*max(max(LabelPairCost))*NumEdge);

NumTS = NumLabel^NumVertex;  %Number of Total States
NumGS =  size(DataStates,2); %Number of Ground States
NumES = NumTS - NumGS; %Number of Excited states

assert(NumGS<=NumTS, 'Too many states \n')
Labels = 1: NumLabel;
assert(sum(sum(ismember(DataStates,Labels)))==NumVertex*NumGS...
    ,'Unqualified states \n')


% GSSetIndex: Index of Ground states
% ESSetIndex: Index of Excited states
GSSetIndex = State2Index(DataStates, NumVertex, NumLabel);
ESSetIndex = 1:NumTS;
States = Index2State(ESSetIndex, NumVertex, NumLabel);
ESSetIndex(:,GSSetIndex)=[];

%Number of parameters
NumParam = NumVertex + NumEdge;

% GSSet = NumParam x NumGS with kth column = [U(S_i);V(S_i,S_j);] where S is
%         the array of labels corresponding to kth excited state
% ESSet = NumParam x NumES with kth column = [U(S_i);V(S_i,S_j);] where S is
%         the array of labels corresponding to kth data state
GSSet = zeros(NumParam,NumGS);
ESSet = zeros(NumParam,NumES);


for k=1:NumVertex
    GSSet(k,:)=LabelSingletCost(States(k,GSSetIndex));
    ESSet(k,:)=LabelSingletCost(States(k,ESSetIndex));
end
for k=1:NumEdge
    i = EdgeMap(k,1);
    j = EdgeMap(k,2);
    temp = (States(j,GSSetIndex)-1)*NumLabel + (States(i,GSSetIndex));
    GSSet(NumVertex +k,:)=LabelPairCost(temp);
    temp = (States(j,ESSetIndex)-1)*NumLabel + (States(i,ESSetIndex));
    ESSet(NumVertex +k,:)=LabelPairCost(temp);
end

Jlb = JLim(1);
Jub = JLim(2);
Hlb = HLim(1);
Hub = HLim(2);

%-------------------------------------------------------------------------
%Matrices for Mixed Integer Linear Programming
%-------------------------------------------------------------------------
% Variables: [ {J_ij}, {H_i}, E_1, {m_i}]
%             J_ij - Edge strength of (ij) connection
%             H_i - Field strength of i-vertex
%             E_1 - Energy of lowest Excited state
%             m_i - {0/1} variable corresponding ith Excited state
%                    m_i = 1 if E_i has lowest energy among excited states
%                    m_i = 0 if E_i otherwise
display('Estimating matrices for MILP')
% Objective function
% Cost = E(1st Ground State) - E_1
f = [GSSet(:,1); -1; sparse(NumES,1)];

% Inequality conditions
%  E(kth excited state) - E_1 + M*q_k <= M for all k
% -E(kth excited state) + E_1         <= 0 for all k
A = sparse([[ESSet',-ones(NumES,1),      M*speye(NumES)]; ...
    [-ESSet',ones(NumES,1),sparse(NumES,NumES)]]);
b = [  M*ones(NumES,1);...
    zeros(NumES,1)];

% Equality conditions
%  E(1st Ground state) - E(kth Ground state) = 0        for all k
%  z_1 + z_2 + ...+ z+k                      = 1
if NumGS==1
    Aeq =sparse( [zeros(1, NumParam),0,ones(1,NumES)]);
    beq = 1;
    
else
    Aeq =sparse( [[(GSSet(:,1)*ones(1,NumGS-1) - GSSet(:,2:end))', ...
        sparse(NumGS-1,1), sparse(NumGS-1,NumES)]; ...
        [zeros(1, NumParam),0,ones(1,NumES)]]);
    beq = [zeros(NumGS-1,1);...
        1];
end

% Bounds
%  JLim(1)<= J_ij <= JLim(2)
%  HLim(1)<=  H_i <= HLim(2)
%      -M <=  E_1 <= M
%       0 <=  m_i <= 1
lb = [Hlb*ones(NumVertex,1);Jlb*ones(NumEdge,1);-M;zeros(NumES,1)];
ub = [Hub*ones(NumVertex,1);Jub*ones(NumEdge,1);M;ones(NumES,1)];

% Integera variables: {m_i}
intcon = NumParam + 1+1: NumParam + 1 + NumES;

%-------------------------------------------------------------------------
% Linear Programming step
%-------------------------------------------------------------------------
display('Starting Linear programming step')
options = optimoptions('intlinprog');
options.ConstraintTolerance = 1e-6;
options.IntegerTolerance = 1e-6;
display('Setting Tolerance (Can be changed internally)')
[x,fval] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,options);

%-------------------------------------------------------------------------
% Post-Processing step
%-------------------------------------------------------------------------
HWeights = x(1: NumVertex);
JWeights = sparse(EdgeMap(:,1),EdgeMap(:,2),x(NumVertex + 1: ...
    NumVertex+ NumEdge),NumVertex,NumVertex);

SuccessFlag = 1;
EnergyGap = -fval;
if fval>=0
    warning('No Minimizer found to replicate Ground state')
    SuccessFlag = 0;
    EnergyGap = 0;
end

fprintf('Found Energy gap of %d\n',-fval);
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