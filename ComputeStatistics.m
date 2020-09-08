function [KLDiv, NLL, BetaRange, Energy, GSSetIndex] = ComputeStatistics(GStates,HWeights, JWeights, NumLabel, BetaRange, LabelPairCost,LabelSingletCost,GroundProbs)
%Function to estimate the Energy parameters in Pott's model given the
%ground states
%Input: 
%   GStates = A matrix of size NumVertices x NumGroundStates  with each 
%               elementin the set {1,2,...,NumLabel}. Each column 
%               represents a ground state
%   HWeights = An array of size NumVertices x 1 representing Field
%              parameters
%   JWeights = An upper triangular sparse matrix with (i,j) elements as the
%              edge weight of (i,j) connection
%   NumLabel = Number of Labels
%Optional Input: 
%   BetaRange = An array of beta values to evaluate KLDivergence
%   LabelPairCost = A symmetric matrix of size NumLabel x NumLabel with
%                   (i,j) elements = V(S_i, S_j)
%   LabelSingletCost = An array of size NumLabel with i^th element = U(S_i)
%   GroundProbs = An array of size NumVertices x NumGroundStates  which
%                   sums to one and has positive entries
%Output: 
%   KLDiv: KLDivergence 
%   NLL: Negative Log-likelihood
%   BetaRange: same as input
%   Energy: Energy of each state
%   GSSetIndex: Index of ground states

display(GStates);
%-------------------------------------------------------------------------
% Setting up optional inputs
%-------------------------------------------------------------------------
NumVertex = size(GStates,1);
NumTS = NumLabel^NumVertex;  %Number of Total States
NumGS =  size(GStates,2); %Number of Ground States

if nargin<8
    % Equally probable
    GroundProbs = ones(1,NumGS)./NumGS;
else
    GroundProbs = GroundProbs(:)';
    assert(size(GroundProbs,1)==1 && size(LabelPairCost,2)==NumGS, ...
    'Incorrect size of GroundProbs vector\n' )
    assert(sum(GroundProbs>0)==NumGS && sum(GroundProbs<=1)==NumGS, ...
    'Incorrect range of GroundProbs vector\n' )
    assert(abs(sum(GroundProbs) - 1)<10*eps, ...
    'Sum of GroundProbs vector is not 1 \n' )
end

if nargin<7
    % U(S_i) is linearly graded between -1 and 1
    LabelSingletCost = linspace(-1,1,NumLabel);
else
    assert(length(LabelSingletCost)==NumLabel,...
        'LabelSingletCost should be a vector of size NumLabel\n');
end

if nargin<6
    % V(S_i, S_j) = 1 for all i != j 
    LabelPairCost = ones(NumLabel) - eye(NumLabel);
else
    assert(size(LabelPairCost,1)==NumLabel && size(LabelPairCost,2)==NumLabel, ...
    'LabelPairCost should be a square matrix of size NLabel\n' )
    LabelPairCost = (LabelPairCost+LabelPairCost')/2; %Turn into symmetric matrix
end

if nargin<5
    % V(S_i, S_j) = 1 for all i != j 
    BetaRange = logspace(-1,2,20);
else
    BetaRange = BetaRange(:).';
    assert(sum(BetaRange>0)==length(BetaRange), ...
    'Incorrect range of GroundProbs vector\n' )
end

assert(NumGS<=NumTS, 'Too many states \n')
Labels = 1: NumLabel;
assert(sum(sum(ismember(GStates,Labels)))==NumVertex*NumGS...
    ,'Unqualified states \n')

%-------------------------------------------------------------------------
% Setting up useful variables
%-------------------------------------------------------------------------

%Matrices for indexing Edges
if istriu(JWeights)
    JWeights = JWeights+JWeights'; %Ensure symmetric JWeights
end
Connection = triu(JWeights~=0); 
[NumEdge,EdgeMap,~] = IndexEdges(NumVertex,Connection);


% GSSetIndex: Index of Ground states
% StateIndex: Index of Excited states
GSSetIndex = State2Index(GStates, NumVertex, NumLabel); 
StateIndex = 1:NumTS;
States = Index2State(StateIndex, NumVertex, NumLabel);


% Energy = 1 x NumTS with kth column = E(S_i,S_j) where S is
%         the array of labels corresponding to kth state 
%         E(S_i,S_j) = H_i U(S_i) + J_ij V(S_i,S_j) 
Energy = zeros(1,NumTS);

for k=1:NumEdge
    i = EdgeMap(k,1);
    j = EdgeMap(k,2);
    temp = (States(j,StateIndex)-1)*NumLabel + (States(i,StateIndex));
    Jval = full(JWeights(i,j));
    Energy = Energy + Jval*LabelPairCost(temp);
end
for k=1:NumVertex
    Energy = Energy + HWeights(k).*LabelSingletCost(States(k,StateIndex));
end

KLDiv = zeros(size(BetaRange)); 
NLL = zeros(size(BetaRange));
for k = 1:length(BetaRange)
   beta = BetaRange(k);
   Probs = exp(-beta*Energy); %Boltzmann distribution
   Probs = Probs./sum(Probs); %Normalization
   KLDiv(k) = - sum(GroundProbs.*log(Probs(GSSetIndex)./GroundProbs)); 
   NLL(k) = - sum(log(Probs(GSSetIndex)));
end


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