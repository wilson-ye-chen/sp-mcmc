function [Mu, S, w] = gmparam_hd_rnd()
% [Mu, S, w] = gmparam_hd_rnd() generates random parameter values for
% a higher-dimensional Gaussian mixture distribution.
%
% Output:
% Mu - 10-by-nDim matrix of mean vectors.
% S  - nDim-by-nDim-by-10 array of covariance matrices.
% w  - 10-by-1 vector of weights.
%
% Date: July 26, 2018

    nMix = 5;
    nDim = 10;
    Mu = unifrnd(-3, 3, nMix, nDim);
    mnDgnl = 2;
    df = 20;
    S = zeros(nDim, nDim, nMix);
    for i = 1:nMix
        S(:, :, i) = wishrnd(eye(nDim) .* mnDgnl ./ df, df);
    end
    w = unifrnd(8, 12, nMix, 1);
    w = w ./ sum(w);
end
