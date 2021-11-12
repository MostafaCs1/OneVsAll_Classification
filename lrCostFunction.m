function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Main Code

h = sigmoid(X*theta);
J = (-1/m)*(y'*log(h) + (1-y)'*log(1-h));
grad = (1/m)*(X'*(h-y));

temp = theta;
temp(1) = 0;

J = J + (lambda/(2*m))*sum(temp.^2);
grad = grad + (lambda/m)*temp;

grad = grad(:);

end
