function stump = buildStump(X, y, weight)
D = size(X, 2); % Dimension

if nargin <= 2
    weight = ones(size(X,1), 1);
end

cellDS = cell(D, 1);
Err = zeros(D, 1);
for i = 1:D
    cellDS{i} = buildOneDStump(X(:,i), y, i, weight);
    Err(i) = cellDS{i}.error;
end
[v, idx] = min(Err);
stump = cellDS{idx};
end
