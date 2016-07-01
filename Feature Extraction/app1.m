% Extract Discrete Wavelet Transform (DWT) Feature
close all; clear; clc;

load dataset_BCIcomp1.mat
Y=y_train;

% Range is 0 to 9 sec
startS=0;
endS=9;
wStep=1;
wRange=9;

X=extractDWT(x_train,startS,endS,wStep,wRange);
T=extractDWT(x_test,startS,endS,wStep,wRange);
save dataDWT.mat X Y T


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


