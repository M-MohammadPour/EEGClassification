function stump = buildOneDStump(x, y, d, w)
[err_1, t_1] = searchThreshold(x, y, w, '>'); % > t_1 -> +1
[err_2, t_2] = searchThreshold(x, y, w, '<'); % < t_2 -> +1
stump = initStump(d);
if err_1 <= err_2
    stump.threshold = t_1;
    stump.error = err_1;
    stump.less = -1;
    stump.more = 1;
else
    stump.threshold = t_2;
    stump.error = err_2;
    stump.less = 1;
    stump.more = -1;
end
end

function [error, thresh] = searchThreshold(x, y, w, sign)
N = length(x);
err_n = zeros(N, 1);
y_predict = zeros(N, 1);
for n=1:N
    switch sign
        case '>'
            idx = logical(x >= x(n));
            y_predict(idx) = 1;
            y_predict(~idx) = -1;
        case '<'
            idx = logical(x < x(n));
            y_predict(idx) = 1;
            y_predict(~idx) = -1;
    end
    err_label = logical(y ~= y_predict);
    %sum(err_label)
    err_n(n) = sum(err_label.*w)/sum(w);
end
[v, idx] = min(err_n);
error = v;
thresh = x(idx);
end
