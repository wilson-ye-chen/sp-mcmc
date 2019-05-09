function [Mu, S, w] = gmparam_trvl(scale)
% [Mu, S, w] = gmparam_trvl(scale) returns the parameter values
% for a two-dimensional uncorrelated Gaussian distribution with
% a given scale.
%
% Input:
% scale - variance for both dimensions.
%
% Output:
% Mu    - 1-by-2 mean vector.
% S     - 2-by-2 matrix of covariance matrices.
% w     - weight = 1.
%
% Date: January 9, 2019

    Mu = [0, 0];
    S = [scale, 0; 0, scale];
    w = 1;
end
