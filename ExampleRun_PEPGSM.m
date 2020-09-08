clear
clc

%Choose number of nodes and labels
NumLabel = 2; NumVertex = 4;
NumGStates = 2;
HLim = [-1 1];
JLim = [-1,1];

%Optional Inputs
LabelSingletCost = linspace(-1,1,NumLabel);
LabelPairCost = ones(NumLabel) - eye(NumLabel);
Connection = triu(ones(NumVertex),1); 

%Calculate Pott's parameter
[HWeights, JWeights, EnergyGap, GStates, SuccessFlag] = ...
    PEPGSM(NumLabel,NumVertex,NumGStates,HLim, JLim, ...
    LabelPairCost,LabelSingletCost,Connection);
%Plot weighted graph
figure(1); 
h1 = PlotGraph(JWeights, HWeights);

%Compute statistics
BetaRange = logspace(-2,2,20);
[KLDiv, NLL, BetaRange, Energy, GSSetIndex] = ComputeStatistics(GStates,HWeights, JWeights, NumLabel,...
    BetaRange, LabelPairCost,LabelSingletCost);

%Plot KL Divegence
figure(2)
h2 = semilogx(BetaRange, KLDiv, 'r-','LineWidth',2);
xlabel('$\beta$','Interpreter','Latex')
ylabel('KL Divergence','Interpreter','Latex');

%Plot Negative Log-likelihood
figure(3)
h3 = semilogx(BetaRange, NLL, 'r-','LineWidth',2);
xlabel('$\beta$','Interpreter','Latex')
ylabel('Negative Log-Likelihood','Interpreter','Latex')
hold on