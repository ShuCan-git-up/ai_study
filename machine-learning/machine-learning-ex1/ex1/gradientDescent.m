function [theta2_vals, theta3_vals, theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta2_vals = zeros(1, num_iters);
theta3_vals = zeros(1, num_iters);
fprintf('在gradientDescent里面输出theta:%f',theta);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % 此处进行梯度下降，一直改进参数theta，从而更加精确的来获取具体的损失值 

    partial1 = 0;
    partial2 = 0;
    for i = 1:m
	xsub = X(i,:);
	ysub = y(i,:);
	partial1 += (dot(transpose(theta), xsub,2) - ysub) * X(i,1);
	partial2 += (dot(transpose(theta), xsub,2) - ysub) * X(i,2);
    end
    theta(1,:) = theta(1,:) - alpha*partial1/m;
    theta(2,:) = theta(2,:) - alpha*partial2/m;	
    theta2_vals(1, iter) =  theta(1,:);
    theta3_vals(1, iter) =  theta(2,:);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
