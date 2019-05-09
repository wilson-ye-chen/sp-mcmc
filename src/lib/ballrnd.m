function X = ballrnd(c, r, n)
% X = ballrnd(c, r, n) generates random numbers uniformly from a ball.
%
% Input:
% c - centre as an 1-by-nDim vector.
% r - radius of the ball.
% n - sample size.
%
% Output:
% X - n-by-nDim matrix of generated points.
%
% Date: July 9, 2018

    nDim = numel(c);
    X = randn(n, nDim);
    X = X ./ repmat(sqrt(sum(X .^ 2, 2)), 1, nDim);
    X = X .* repmat(rand(n, 1) .^ (1 ./ nDim), 1, nDim);
    X = X .* r + repmat(c, n, 1);
end
