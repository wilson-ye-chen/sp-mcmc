function Y = logistic(X, k)
% Y = logistic(X, k) implements the standard logistic function
% parameterised by the growth rate k.
%
% Date: January 11, 2019

    Y = 1 ./ (1 + exp(-k .* X));
end
