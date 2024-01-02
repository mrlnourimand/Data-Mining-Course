clear
clc
% Import the data from the Excel file
[~, ~, raw] = xlsread('bloodp.xlsx');

% Use import to read data from Excel
%data = importdata('bloodp.xlsx');

% Question 2
% Extract systolic (sbp) and diastolic (dbp) blood pressure values
sbp = cell2mat(raw(2:end, 1));
dbp = cell2mat(raw(2:end, 2));

% calculate the mean value
sbp_mean = round(mean(sbp(sbp > 0), 'omitnan'))
dbp_mean = round(mean(dbp(dbp > 0), 'omitnan'))

% Replace zero values with mean values
sbp(sbp == 0) = sbp_mean;
dbp(dbp == 0) = dbp_mean;

% Replace missing values with mean values
sbp(isnan(sbp)) = sbp_mean;
dbp(isnan(dbp)) = dbp_mean;

% Correct erroneous values
sbp(sbp < 80) = sbp(sbp < 80) * 10;
dbp(dbp < 40) = dbp(dbp < 40) * 10;

% Remove values that are impossible

% first find the problematic rows in
% both sbp and dbp
logical_array = (sbp > 300 | dbp > 160);

% then remove the rows which fulfill the condition(sbp > 300 or dbp > 160)
sbp(logical_array == 1) = [];
dbp(logical_array == 1) = [];

% corrected data
corrected_data = [sbp, dbp];


% Question 2
% create the observation matrix O
O = [ones(size(sbp)), sbp, dbp];

% select y=dbp and X=[1 sdb]
y = O(:, 3);
X = O(:, [1, 2]);

% compute coefficients manually
coefficients = (X' * X) \ (X' * y);

disp('Coefficients (Manual Computation):');
disp(coefficients);


% use the regress function
coefficients_regress = regress(y, X);

disp('Coefficients (Using regress function):');
disp(coefficients_regress);



% Question 3
% loading data given
S = {'word1', 'word2', 'word3', 'word4', 'word5'};
Fo = [15, 7, 6, 11, 4];
Nw = 500;

Fo1 = [1, 4, 3, 3, 6];
Nw1 = 200;

Fo2 = [20, 1, 5, 16, 9];
Nw2 = 210;

% normalize word occurrences
normalized_Fo = Fo / Nw;
normalized_Fo1 = Fo1 / Nw1;
normalized_Fo2 = Fo2 / Nw2;

% calculate the cosine distance
% dot = dot product of two vectors
% norm = Euclidean norm (magnitude) of a vector
cosine_distance_1 = 1 - dot(normalized_Fo, normalized_Fo1) / (norm(normalized_Fo) * norm(normalized_Fo1));
cosine_distance_2 = 1 - dot(normalized_Fo, normalized_Fo2) / (norm(normalized_Fo) * norm(normalized_Fo2));

disp('Cosine Distance between Reference and Document 1:');
disp(cosine_distance_1);

disp('Cosine Distance between Reference and Document 2:');
disp(cosine_distance_2);


% Question 4
% Load the dataset from the Excel file
power_data = xlsread('Tetuan City power consumption.csv');

%"Binarizing" a sample in the context of this code means converting 
% its values into binary values based on a threshold. Specifically, it 
% involves comparing each value in the sample to the mean value of the 
% corresponding variable in the dataset. If the value is less than the
% mean, it becomes 0; otherwise, it becomes 1. This process results in
% a binary representation of the sample.

% Binarize all variables
mean_values = mean(power_data);
binarized_data = power_data >= mean_values;

% Sample data
s = [0, 1, 0, 0, 0, 0, 0, 0];

% Binarize the sample
binarized_s = s >= mean_values;

% Calculate Hamming distance
% (the variable names are stored in the first row of the Excel sheet)
hamming_distances = sum(binarized_data ~= binarized_s, 2);

% Find the index of the nearest neighbor
% The tilde (~) is used as a placeholder to indicate that we are not 
% interested in the actual minimum value, only its index
[~, nearest_neighbor_index] = min(hamming_distances);

% Display the nearest neighbor
nearest_neighbor = power_data(nearest_neighbor_index, :);
disp('Nearest Neighbor:');
disp(nearest_neighbor);


% Question 5
% Correlation for binarized_data, method 1
R = corrcoef(binarized_data)
highest_corr = max(R)

% method 2
% calculates the number of variables (columns) in the binarized dataset
num_variables = size(binarized_data, 2);
binary_correlation = zeros(num_variables);

for i = 1:num_variables
    for j = 1:num_variables
        % Calculate binary correlation using XOR
        binary_correlation(i, j) = sum(bitxor(binarized_data(:, i), binarized_data(:, j)));
    end
end

% Display binary correlation matrix
disp('Binary Correlation Matrix:');
disp(binary_correlation);

% Identify variables with the highest correlation, know the column and row
[max_correlation, ind] = max(binary_correlation(:));
[row, col] = ind2sub(size(binary_correlation), ind);

disp('Variables with the Highest Binary Correlation:');
disp(['Variable ' num2str(row) ' and Variable ' num2str(col) ' with Correlation ' num2str(max_correlation)]);

% method 3, where E should be the mean. I'm not sure x and y should be the
% same here or not...
%R3 = sum((x - E).*(y - E)) ./ (sqrt(sum((x-E).^2)) .* sqrt(sum((y-E).^2)));
