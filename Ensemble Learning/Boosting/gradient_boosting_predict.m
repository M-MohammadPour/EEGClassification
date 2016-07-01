% Practicum, Task #3, 'Compositions of algorithms'.
%
% FUNCTION:
% [prediction, err] = gradient_boosting_predict (model, X, y)
%
% DESCRIPTION:
% This function use the composition of algorithms, trained with gradient 
% boosting method, for prediction.
%
% INPUT:
% X --- matrix of objects, N x K double matrix, N --- number of objects, 
%       K --- number of features.
% y --- vector of answers, N x 1 double vector, N --- number of objects. y
%       can have only two values --- +1 and -1.
% model --- trained composition.
%
% OUTPUT:
% prediction --- vector of predicted answers, N x 1 double vector.
% error --- the ratio of number of correct answers to number of objects on
%           each iteration, num_iterations x 1 vector
%
% AUTHOR: 
% Murat Apishev (great-mel@yandex.ru)
%

function [prediction, err] = gradient_boosting_predict (model, X, y)

    num_iterations = length(model.weights);
    no_objects = length(y);
    pred_prediction = zeros([no_objects num_iterations]);

    for alg = 1 : num_iterations
        value = zeros([no_objects 1]) + model.b_0;
        for i = 1 : alg
            if strcmp(model.algorithm, 'epsilon_svr')
                value = value + svmpredict(y, X, model.models{i}) * model.weights(i);
            elseif strcmp(model.algorithm, 'regression_tree')
                value = value + predict(model.models{i}, X) * model.weights(i);
            end
        end
        pred_prediction(:,alg) = value;
    end
    prediction = pred_prediction(:,end);
    err = zeros([num_iterations 1]);
    if strcmp(model.loss, 'absolute')
        temp = (bsxfun(@minus, pred_prediction, y));
        err = abs(sum(temp)) / no_objects;
    elseif strcmp(model.loss, 'logistic')
        prediction = sign(prediction);
        temp = (bsxfun(@eq, sign(pred_prediction), y));
        err = sum(temp == 0) / no_objects;
    end
    if size(err, 1) == 1
        err = err';
    end
end