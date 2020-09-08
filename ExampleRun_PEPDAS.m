clear
clc

%Choose number of nodes and labels
NumLabel = 2; NumVertex = 6;

%Define ground states
NumGStates = 3;
GStates = randi([1 NumLabel],[NumVertex NumGStates]);

%Limits for parameters
HLim = [-1 1];
JLim = [-1,1];

%Optional Inputs
    %Cost paramters correspond to 0/1 - Ising model
LabelSingletCost = linspace(0,1,NumLabel);
LabelPairCost = [0,0;0,1]; 
Connection = triu(ones(NumVertex),1); %Fully connected graph

%Calculate Pott's parameter
[HWeights, JWeights, EnergyGap,SuccessFlag] = ...
        PEPDAS(NumLabel,GStates,HLim, JLim, ...
        LabelPairCost,LabelSingletCost,Connection); 
%Plot weighted graph
figure(1)
h1 = PlotGraph(JWeights, HWeights);

%Compute statistics
BetaRange = logspace(-5,2,50);
[KLDiv, NLL, BetaRange, Energy, GSSetIndex] = ComputeStatistics(GStates,HWeights, JWeights, NumLabel,...
    BetaRange, LabelPairCost,LabelSingletCost);

%Plot KL Divegence
figure(2)
h2 = semilogx(BetaRange, KLDiv, 'r-','LineWidth',2);
xlabel('$\beta$','Interpreter','Latex')
ylabel('KL Divergence','Interpreter','Latex')

%Plot Negative Log-likelihood
figure(3)
h3 = semilogx(BetaRange, NLL, 'r-','LineWidth',2);
xlabel('$\beta$','Interpreter','Latex')
ylabel('Negative Log-Likelihood','Interpreter','Latex')
hold on
