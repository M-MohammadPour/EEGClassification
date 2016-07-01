% Practicum, Task #3, 'Compositions of algorithms'.
%
% FUNCTION:
% [model] = gradient_boosting_train (X, y, num_iterations, base_algorithm, loss, ...
%                               param_name1, param_value1, param_name2, param_value2, ...
%                               param_name3, param_value3, param_name3, param_value3)
% 
% DESCRIPTION:
% This function train the composition of algorithms using gradient boosting method.
%
% INPUT:
% X --- matrix of objects, N x K double matrix, N --- number of objects, 
%       K --- number of features.
% y --- vector of answers, N x 1 double vector, N --- number of objects. y
%       can have only two values --- +1 and -1 in case of classification
%       and all possible double values in case of regression.
% num_iterations --- the number ob algorithms in composition, scalar.
% base_algorithm --- the base algorithm, string. Can have one of two
%                    values: 'regression_tree' or 'epsilon_svr'.
% loss --- the loss function, string. Can have one of two values: 
%          'logistic' (for classification) or 'absolute' (for regression).
% param_name1 --- learning rate, scalar.
% param_name2 --- parameter of base_algorithm. For 'regression_tree' it 
%                 is a 'min_parent' --- min number of objects in the leaf of 
%                 regression tree. For 'epsilon_svr' it is 'epsilon' parameter.
% param_name3 --- parameter, that exists only for 'epsilon_svr', it is a 
%                 'gamma' parameter.
% param_name4 --- parameter, that exists only for 'epsilon_svr', it is a 
%                 'C' parameter.
% param_value1, param_value2, param_value3, param_value4 --- values of 
%                                       corresponding parametres, scalar.
% OUTPUT:
% model --- trained composition, structure with three fields
%       - b_0 --- the base of composition, scalar
%       - weights --- double array of weights, 1 x num_iterations
%       - models --- cell array with trained models, 1 x num_iterations
%       - algorithm --- string, 'epsilon_svr' or 'regression_tree'
%       - loss --- loss parameter (from INPUT).
% 
% AUTHOR: 
% Murat Apishev (great-mel@yandex.ru)
%

function [model] = gradient_boosting_train (X, y, num_iterations, base_algorithm, loss, ...
                                    param_name1, param_value1, param_name2, param_value2, ...
                                    param_name3, param_value3, param_name4, param_value4)
                                
    no_objects = size(X, 1);
    if ~strcmp(base_algorithm, 'epsilon_svr') && ~strcmp(base_algorithm, 'regression_tree')
        error('Incorrect type of algorithm!')
    end
    
    if strcmp(loss, 'logistic')
        loss_function = @(a, b) log(1 + exp(-a .* b));        
        grad_a_loss_function = @(a, b) -b .* exp(-a .* b) ./ ((1 + exp(-a .* b)));
    elseif strcmp(loss, 'absolute')
        loss_function = @(a, b) abs(a - b);
        grad_a_loss_function = @(a, b) -sign(b - a);
    else
        error('Incorrect type of loss function!');
    end
    
    func = @(c) sum(loss_function(y, c));
    b_0 = fminsearch(func, 0);
    model.b_0 = b_0;
    model.algorithm = base_algorithm;
    model.models = cell([1 num_iterations]);
    model.loss = loss;
    % the length of model is number of finite weights, not number of models!
    model.weights = repmat(+Inf, 1, num_iterations);
    
    z = zeros([no_objects 1]) + b_0;
    delta = zeros([no_objects 1]);
    
    for iter = 1 : num_iterations
        for obj = 1 : no_objects
            delta(obj) = -grad_a_loss_function(z(obj), y(obj));
        end
        
        if strcmp(base_algorithm, 'epsilon_svr')
            model.models{iter} = svmtrain(delta, X, [' -s 3 -g ', num2str(param_value3), ...
                            ' -c ', num2str(param_value4), ' -e ', num2str(param_value2)]);
            value = svmpredict(y, X, model.models{iter});
        elseif strcmp(base_algorithm, 'regression_tree')          
            model.models{iter} = RegressionTree.fit(X, delta, 'minparent', param_value2);
            value = predict(model.models{iter}, X);
        end
        
        func = @(g) sum(loss_function(z + g * value, y));
        model.weights(iter) = fminsearch(func, 0);
        z = z + model.weights(iter) * value * param_value1;
    end
end