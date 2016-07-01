function abClassifier = initAdaBoost(N)
abClassifier.nWC = 0;
abClassifier.WeakClas = cell(N,1);
abClassifier.Weight = zeros(N,1);
abClassifier.trnErr = zeros(N, 1);
abClassifier.tstErr = zeros(N, 1);
abClassifier.hasTestData = false;
end
