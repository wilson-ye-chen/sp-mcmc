function p = fp_gaussmix(X, Mu, S, w)
% p = fp_gaussmix(X, Mu, S, w) evaluates the density function of a
% Gaussian mixture distribution.
%
% Input:
% X  - nObs-by-nDim matrix of points.
% Mu - nMix-by-nDim matrix of mean vectors.
% S  - nDim-by-nDim-by-nMix array of covariance matrices.
% w  - nMix-by-1 vector of weights.
%
% Output:
% p  - nObs-by-1 vector of density values.
%
% Date: December 14, 2017

    [nObs, nDim] = size(X);
    nMix = size(Mu, 1);

    Wp = zeros(nObs, nMix);
    T = zeros(nObs, nDim, nMix);
    for i = 1:nMix
        c = sqrt(det(2 .* pi .* S(:, :, i)));
        Y = X - repmat(Mu(i, :), nObs, 1);
        SInvY = S(:, :, i) \ Y';
        Wp(:, i) = w(i) .* exp(-0.5 .* diag(Y * SInvY)) ./ c;
    end
    p = sum(Wp, 2);
end
