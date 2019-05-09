function [Mu, S, w] = gmparam_2c(nDim, scale)
% [Mu, S, w] = gmparam_2c(nDim, scale) returns the parameter values for
% a two-component nDim-dimensional Gaussian mixture distribution.
%
% Input:
% nDim  - number of dimensions (> 0).
% scale - scale of each Gaussian (> 0), e.g., 0.5 or 1.5.
%
% Output:
% Mu    - 2-by-nDim matrix of mean vectors.
% S     - nDim-by-nDim-by-2 array of covariance matrices.
% w     - 2-by-1 vector of weights.
%
% Date: January 8, 2019

    v1 = ones(1, nDim);
    Mu = [-v1; v1];
    S = repmat(scale .* eye(nDim), 1, 1, 2);
    w = [0.5; 0.5];
end
