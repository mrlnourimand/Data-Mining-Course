% Data Mining Exercise, Week 5
clc;
clear;

% Load the Iris dataset
iris_data = dlmread('Iris.txt');

% Extract relevant data
features = iris_data(:, 2:5);  % Columns 2 to 5 are real data
classes = iris_data(:, 6);      % Class column

% Run PCA
% coeff contains the principal component coefficients.
% score contains the principal component scores.
% latent contains the eigenvalues of the covariance matrix.
% explained contains the percentage of variance explained by each principal component.
[coeff, score, latent, ~, explained] = pca(features);

% Display the percentage of variance explained by the first two principal components
fprintf('Percentage of variance explained by the first component: %.2f%%\n', explained(1));
fprintf('Percentage of variance explained by the second component: %.2f%%\n', explained(2));


% Question 2

% k 3:10 (number of nearest neighbors)
k_values = 3:10;

% Initialize a cell array to store feature rankings for each k
feature_rankings_cell = cell(1, length(k_values));

% Perform ReliefF algorithm for different values of k
for k_idx = 1:length(k_values)
    k = k_values(k_idx);

    % Use ReliefF algorithm
    [ranked_features, weights] = relieff(features, classes, k);

    % Store the feature rankings for this k
    feature_rankings_cell{k_idx} = ranked_features;

    % Display the ordered importance of variables for the current k
    fprintf('Ordered Importance of Variables for k = %d:\n', k);
    disp(ranked_features);
    fprintf('\n');
end

% Analyze the effect of the number of nearest neighbors on feature rankings
for k_idx = 1:length(k_values)
    k = k_values(k_idx);
    fprintf('Feature rankings for k = %d:\n', k);
    disp(feature_rankings_cell{k_idx});
    fprintf('\n');
end


% Question 3

% Separate training and test data
training_data = [];
training_labels = [];
test_data = [];
test_labels = [];

for i = 1:3
    class_data = features(classes == i, :);
    
    % Training data: First 40 cases
    training_data = [training_data; class_data(1:40, :)];
    training_labels = [training_labels; repmat(i, 40, 1)];

    % Test data: Remaining 10 cases
    test_data = [test_data; class_data(41:end, :)];
    test_labels = [test_labels; repmat(i, 10, 1)];
end
% Train Naïve Bayesian classifier
nb_model = fitcnb(training_data, training_labels);

% Predict on the test set
predicted_labels = predict(nb_model, test_data);

% Display the confusion matrix
confusionMat = confusionmat(test_labels, predicted_labels);
disp('Confusion Matrix:');
disp(confusionMat);


% Question 4
% use the first two principal components of PCA
selected_features = training_data * coeff(:, 1:2);

% Train Naïve Bayesian classifier
nb_model = fitcnb(selected_features, training_labels);

% Apply PCA transformation to test data
test_data_pca = test_data * coeff(:, 1:2);

% Predict on the test set
predicted_labels = predict(nb_model, test_data_pca);

% Display the confusion matrix
confusionMat = confusionmat(test_labels, predicted_labels);
disp('Confusion Matrix:');
disp(confusionMat);



% Question 5

