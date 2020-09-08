function h = PlotGraph(JWeights, HWeights)
%Function to plot a weighted graph
%Input: 
%   Jweights - A square matrix (NxN) of edge weights
%   HWeights - An array (Nx1) of node weights
%Output: 
%   h - Plot handle
Connection = JWeights; 
G = graph(Connection,'upper'); 

NumVertex = length(HWeights);
NodeLabel = cell(NumVertex,1); 
for i=1:NumVertex
    NodeLabel(i) = {sprintf('H%i= %.2f',i,HWeights(i))};
end
h = plot(G,'EdgeLabel',G.Edges.Weight,'LineWidth',2,...
    'NodeLabel',NodeLabel);
h.Marker = 'o';
h.NodeColor = 'r';
h.MarkerSize = 7;
set(gca,'XTick',[], 'YTick', [])
daspect([1 1 1])
set(gca,'color','none')
set(gca,'Visible','off')

end