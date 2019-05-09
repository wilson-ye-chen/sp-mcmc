function s = ksd(X, D, k)
% s = ksd(X, D, k) generates a cumulative sequence of KSD values.
%
% Input:
% X - nObs-by-nDim matrix of nDim-dimensional points.
% D - nIter-by-nDim matrix of scores at X.
% k - symbolic expression of the kernel k(a,b), where a and b are 1-
%     by-nDim row vectors. It is important that the argument names are
%     literally "a" and "b".
%
% Output:
% s - column vector of cumulative KSD values based on the order of X.
%
% Date: August 1, 2018

    K0 = vechinv(stein_kmat(X, D, k));
    ps = sum(K0, 1)' .* 2 - diag(K0);
    s = sqrt(cumsum(ps)) ./ [1:numel(ps)]';
end
