%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Cost and Gradient descent ===================

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
%theta = [-3.6303;1.1664]

% Some gradient descent settings
iterations = 1500;
alpha = 0.01; 

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [-3.6303;1.1664]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta2_vals = [];
theta3_vals = [];
% 这是梯度下降一步一步根据偏导数慢慢找出来的最优解，可以画在等高图上一步一步连接起来找到结果
[theta2_vals, theta3_vals,theta, J_history] = gradientDescent(X, y, [-10;10], alpha, iterations);
fprintf('输出theta2在迭代过程中所有的值 \n')
theta2_vals
fprintf('输出theta3在迭代过程中所有的值 \n')
theta3_vals
fprintf('输出迭代过后最终的theta值 \n')
theta
fprintf('输出迭代过后最终的J_history值 \n')
J_history

figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
 fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
 %pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
 %fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
%linspace(x1,x2,N) 创建一个N个元素的<向量>1xn 范围在x1,x2之间
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
%c 此处直接给出一个全局损失函数图，从中找到最优解
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

%J_vals = zeros(length(theta0_vals), length(theta1_vals));
%for i = 1:length(theta0_vals)
%	J_vals(i,i) = computeCost(X,y,[theta0_vals(i); theta1_vals(i)])
%end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
% 网状图绘制：surf
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
% logspace(x1,x2,N) 创建一个N个元素的向量,指数分布与10的x1次方和10的x2次方之间
% 绘制等高线：contour
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
for i = 1:length(theta2_vals)
    plot(theta2_vals(i), theta3_vals(i), 'rx', 'MarkerSize', 1, 'LineWidth', 1);
end
