close all; clear;clc;

load dataDWT.mat;
%load dataAR.mat;
%load dataCSP.mat;
%load dataPSD.mat;
Y(Y==2) = -1;

y=Y;

num_iterations=30;
base_algorithm= 'regression_tree';
min_parent = floor(size(X, 1) / 5) * 4 - 1;
loss='logistic';
learning_rate=1;

no_objects = length(y);
fifth = floor(no_objects / 5);
error_train = zeros([num_iterations 1]);
error_test = zeros([num_iterations 1]);

for fold = 1 : 5
    mask_test = zeros([1 no_objects]);
    mask_test((fold - 1) * fifth + 1 : fifth * fold) = 1;
    mask_test = logical(mask_test);
    train_set = X(~mask_test,:);
    train_ans = y(~mask_test);
    test_set = X(mask_test,:);
    test_ans = y(mask_test);
    model = gradient_boosting_train(train_set, train_ans, num_iterations, ...
        base_algorithm, loss, 'learning_rate', learning_rate, ...
        'min_parent', min_parent);
    [~, error_train_loop] = gradient_boosting_predict(model, train_set, train_ans);
    [~, error_test_loop] = gradient_boosting_predict(model, test_set, test_ans);
    error_train = error_train + error_train_loop;
    error_test = error_test + error_test_loop;
end
error_train = error_train / 5;
error_test = error_test / 5;

hold on;
xlabel('Number of iterations');
ylabel('Error');
title('Gradient boosting - PSD');
plot(1 : num_iterations, error_train, 'linewidth', 2);


