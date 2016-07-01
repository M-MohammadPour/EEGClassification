% Practicum, Task #3, 'Compositions of algorithms'.
%
% FUNCTION:
% [model] = bagging_train (X, y, num_iterations, base_algorithm, ...
%                   param_name1, param_value1, param_name2, param_value2)
%
% DESCRIPTION:
% This function train the composition of algorithms using bagging method.
%
% INPUT:
% X --- matrix of objects, N x K double matrix, N --- number of objects, 
%       K --- number of features.
% y --- vector of answers, N x 1 double vector, N --- number of objects. y
%       can have only two values --- +1 and -1.
% num_iterations --- the number ob algorithms in composition, scalar.
% base_algorithm --- the base algorithm, string. Can have one of two
%                    values: 'classification_tree' or 'svm'.
% param_name1 --- parameter of base_algorithm. For 'classification_tree' it 
%            is a 'min_parent' --- min number of objects in the leaf of 
%            classification tree. For 'svm' it is 'gamma' parameter.
% param_name2 --- parameter, that exists only for 'svm', it is a 'C' 
%                 parameter.
% param_value1, param_value2 --- values of corresponding parametres,
%                                scalar.
% OUTPUT:
% model --- trained composition, structure with two fields
%       - models --- cell array with trained models
%       - algorithm --- string, 'svm' or 'classification_tree'
%
% AUTHOR: 
% Murat Apishev (great-mel@yandex.ru)
%

function [model] = bagging_train (X, y, num_iterations, base_algorithm, ...
                        param_name1, param_value1, param_name2, param_value2)
        
    no_objects = size(X, 1);
    models = cell([1 num_iterations]);
                    
    if strcmp(base_algorithm, 'svm')
        if ~strcmp(param_name1, 'gamma')
            temp = param_value1;
            param_value1 = param_value2;
            param_value2 = temp;
        end

        for iter = 1 : num_iterations
            indices = randi(no_objects, 1, no_objects);
            indices = unique(indices);
            models{iter} = svmtrain(y(indices), X(indices,:), ...
                [' -g ', num2str(param_value1), ' -c ', num2str(param_value2)]);
        end
    elseif strcmp(base_algorithm, 'classification_tree')
        for iter = 1 : num_iterations
            indices = randi(no_objects, 1, no_objects);
            indices = unique(indices);
            if (param_value1 > length(indices))
                value = length(indices);
            else
                value = param_value1;
            end
            models{iter} = ClassificationTree.fit(X(indices,:), y(indices), 'MinParent', value);
        end        
    else
        error('Incorrect type of algorithm!');
    end
    
    model.models = models;
    model.algorithm = base_algorithm;
end