function d = pdst(X)
% d = pdst(X) returns the Euclidean distances between pairs of
% observations in X.
%
% Input:
% X - nObs-by-nDim matrix of observations.
%
% Output:
% d - (nObs-choose-2)-by-1 vector of pair-wise distances.
%
% Date: January 17, 2019

    nObs = size(X, 1);
    nElm = (nObs .^ 2 + nObs) ./ 2;
    [r, c] = i2rc(1:nElm);
    abv = r ~= c;
    rAbv = r(abv);
    cAbv = c(abv);
    d = sqrt(sum((X(rAbv, :) - X(cAbv, :)) .^ 2, 2));
end
