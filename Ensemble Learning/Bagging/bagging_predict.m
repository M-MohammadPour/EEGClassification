% Practicum, Task #3, 'Compositions of algorithms'.
%
% FUNCTION:
% [prediction, err] = bagging_predict (model, X, y)
%
% DESCRIPTION:
% This function use the composition of algorithms, trained with bagging 
% method, for prediction.
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

function [prediction, err] = bagging_predict (model, X, y)

    num_iterations = length(model.models);
    no_objects = length(y);
    pred_prediction = zeros([no_objects num_iterations]);
    err = zeros([num_iterations 1]);
    
    if strcmp(model.algorithm, 'svm')
        for alg = 1 : num_iterations
            pred_prediction(:,alg) = svmpredict(y, X, model.models{alg});
            func = @(i) find_max(pred_prediction(i,:));
            prediction = arrayfun(func, 1 : no_objects)';
            err(alg) = sum(prediction ~= y) / no_objects;
        end
    elseif strcmp(model.algorithm, 'classification_tree')
        for alg = 1 : num_iterations
            pred_prediction(:,alg) = predict(model.models{alg}, X);
            func = @(i) find_max(pred_prediction(i,:));
            prediction = arrayfun(func, 1 : no_objects)';
            err(alg) = sum(prediction ~= y) / no_objects;            
        end
    else
        error('Incorrect type of algorithm!');
    end    
end

function [result] = find_max (vector)

    if sum(vector == -1) > sum(vector == +1)
        result = -1;
    else
        result = +1;
    end
end