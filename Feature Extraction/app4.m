% Extract Common Spatial Pattern (CSP) Feature
close all; clear; clc;

load dataset_BCIcomp1.mat

EEGSignals.x=x_train;
EEGSignals.y=y_train;
Y=y_train;

classLabels = unique(EEGSignals.y); 
CSPMatrix = learnCSP(EEGSignals,classLabels);
nbFilterPairs = 1;

X = extractCSP(EEGSignals, CSPMatrix, nbFilterPairs);  
EEGSignals.x=x_test;
T = extractCSP(EEGSignals, CSPMatrix, nbFilterPairs);  

save dataCSP.mat X Y T


color_L = [0 102 255] ./ 255;
color_R = [255, 0, 102] ./ 255;

pos = find(Y==1);
plot(X(pos,1),X(pos,2),'x','Color',color_L,'LineWidth',2);

hold on
pos = find(Y==2);
plot(X(pos,1),X(pos,2),'o','Color',color_R,'LineWidth',2);

legend('Left Hand','Right Hand')
xlabel('C3','fontweight','bold')
ylabel('C4','fontweight','bold')

