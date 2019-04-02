function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

theta = pinv(transpose(X) * X) * transpose(X) * y

% -------------------------------------------------------------

fprintf('输出预测的面积在1650,三房的价格:')
transpose(theta) * [1;1650;3]

% ============================================================

end
