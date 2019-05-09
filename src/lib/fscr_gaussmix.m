function [D, p] = fscr_gaussmix(X, Mu, S, w)
% [D, p] = fscr_gaussmix(X, Mu, S, w) returns scores for a Gaussian
% mixture distribution.
%
% Input:
% X  - nObs-by-nDim matrix of points.
% Mu - nMix-by-nDim matrix of mean vectors.
% S  - nDim-by-nDim-by-nMix array of covariance matrices.
% w  - nMix-by-1 vector of weights.
%
% Output:
% D  - nObs-by-nDim matrix of scores.
% p  - nObs-by-1 vector of density values.
%
% Date: December 15, 2017

    [nObs, nDim] = size(X);
    nMix = size(Mu, 1);

    Wp = zeros(nObs, nMix);
    T = zeros(nObs, nDim, nMix);
    for i = 1:nMix
        c = sqrt(det(2 .* pi .* S(:, :, i)));
        Y = X - repmat(Mu(i, :), nObs, 1);
        SInvY = S(:, :, i) \ Y';
        Wp(:, i) = w(i) .* exp(-0.5 .* diag(Y * SInvY)) ./ c;
        T(:, :, i) = repmat(Wp(:, i), 1, nDim) .* SInvY';
    end
    p = sum(Wp, 2);
    D = -sum(T, 3) ./ p;
end
