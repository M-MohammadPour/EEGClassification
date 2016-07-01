function [Label, Err] = predAdaBoost(abClassifier, X, Y)
N = size(X, 1);

if nargin < 3
    Y = [];
end

M = abClassifier.nWC;
LabM = zeros(N, M);
for i = 1:M
    LabM(:,i) = abClassifier.Weight(i)*predStump(X, abClassifier.WeakClas{i});
end

% 
Label = zeros(N, 1);
LabM = sum(LabM, 2);
idx = logical(LabM > 0);
Label(idx) = 1;
Label(~idx) = -1;

% 
if ~isempty(Y)
    Err = logical(Label ~= Y);
    Err = sum(Err)/N;
end
end
