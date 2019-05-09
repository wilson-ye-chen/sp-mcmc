function X = gaussmix_rnd(Mu, S, w, nObs)
% X = gaussmix_rnd(Mu, S, w, nObs) generates independent draws from a
% Gaussian mixture distribution.
%
% Input:
% Mu   - nMix-by-nDim matrix of mean vectors.
% S    - nDim-by-nDim-by-nMix array of covariance matrices.
% w    - vector of weights.
% nObs - number of random draws.
%
% Output:
% X    - nObs-by-nDim matrix of points.
%
% Date: January 3, 2018

    nDim = size(Mu, 2);
    X = zeros(nObs, nDim);
    w = w(:)';
    for i = 1:nObs
        j = logical(mnrnd(1, w));
        X(i, :) = mvnrnd(Mu(j, :), S(:, :, j));
    end
end
