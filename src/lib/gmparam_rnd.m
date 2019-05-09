function [Mu, S, w] = gmparam_rnd()
% [Mu, S, w] = gmparam_rnd() generates random parameter values for
% a bivariate Gaussian mixture distribution.
%
% Output:
% Mu - 10-by-2 matrix of mean vectors.
% S  - 2-by-2-by-10 array of covariance matrices.
% w  - 10-by-1 vector of weights.
%
% Date: January 2, 2018

    nMix = 10;
    nDim = 2;
    Mu = unifrnd(-4, 4, nMix, nDim);
    S = zeros(nDim, nDim, nMix);
    for i = 1:nMix
        S(:, :, i) = wishrnd(eye(nDim) .* 0.2, 8);
    end
    w = unifrnd(8, 12, nMix, 1);
    w = w ./ sum(w);
end
