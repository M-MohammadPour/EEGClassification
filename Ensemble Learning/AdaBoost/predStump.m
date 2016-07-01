% Make prediction based on a decision stump

function label = predStump(X, stump)
N = size(X, 1);
x = X(:, stump.dim);
idx = logical(x >= stump.threshold); % N x 1
label = zeros(N, 1);
label(idx) = stump.more;
label(~idx) = stump.less;
end
