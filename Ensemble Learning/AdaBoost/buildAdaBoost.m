function abClassifier = buildAdaBoost(trnX, trnY, iter, tstX, tstY)
if nargin < 4
    tstX = [];
    tstY = [];
end
abClassifier = initAdaBoost(iter);

N = size(trnX, 1); % Number of training samples
sampleWeight = repmat(1/N, N, 1);

for i = 1:iter
    weakClassifier = buildStump(trnX, trnY, sampleWeight);
    abClassifier.WeakClas{i} = weakClassifier;
    abClassifier.nWC = i;
    % Compute the weight of this classifier
    abClassifier.Weight(i) = 0.5*log((1-weakClassifier.error)/weakClassifier.error);
    % Update sample weight
    label = predStump(trnX, weakClassifier);
    tmpSampleWeight = -1*abClassifier.Weight(i)*(trnY.*label); % N x 1
    tmpSampleWeight = sampleWeight.*exp(tmpSampleWeight); % N x 1
    sampleWeight = tmpSampleWeight./sum(tmpSampleWeight); % Normalized
    
    % Predict on training data
    [ttt, abClassifier.trnErr(i)] = predAdaBoost(abClassifier, trnX, trnY);
    % Predict on test data
    if ~isempty(tstY)
        abClassifier.hasTestData = true;
        [ttt, abClassifier.tstErr(i)] = predAdaBoost(abClassifier, tstX, tstY);
    end
    % fprintf('\tIteration %d, Training error %f\n', i, abClassifier.trnErr(i));
end
end
