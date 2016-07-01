close all; clear; clc;

load dataDWT.mat;
%load dataCSP.mat;
%load dataAR.mat;
%load dataPSD.mat;

Y(Y==2) = -1;

N=size(X,1);
trnX = X(1:N, :);
trnY = Y(1:N);

iter = 30;
abClassifier = initAdaBoost(iter);

N = size(trnX, 1); % Number of training samples
sampleWeight = repmat(1/N, N, 1);

for t = 1:iter
    weakClassifier = buildStump(trnX, trnY, sampleWeight);

    abClassifier.WeakClas{t} = weakClassifier;
    abClassifier.nWC = t;
    % Compute the weight of this classifier
    abClassifier.Weight(t) = 0.5*log((1-weakClassifier.error)/weakClassifier.error);
    weakClassifier.error
    % Update sample weight
    label = predStump(trnX, weakClassifier);
    tmpSampleWeight = -1*abClassifier.Weight(t)*(trnY.*label); % N x 1
    tmpSampleWeight = sampleWeight.*exp(tmpSampleWeight); % N x 1
    
    sampleWeight = tmpSampleWeight./sum(tmpSampleWeight); % Normalized
    
    % Predict on training data
    [ttt, abClassifier.trnErr(t)] = predAdaBoost(abClassifier, trnX, trnY);
    
    fprintf('\tIteration %d, Training error %f\n', t, abClassifier.trnErr(t));
end

trnError = abClassifier.trnErr;
plot(1:iter, trnError);

