function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(iterations, 1);

for iter = 1:iterations
    delta=1/m*(X'*X*theta-X'*y);
    theta=theta-alpha.*delta;
    J_history(iterations) = computeCost(X, y, theta);
end
J_history;
end