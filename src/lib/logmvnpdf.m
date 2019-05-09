function y = logmvnpdf(x, mu, S)
% y = logmvnpdf(x, mu, S) returns the value of the log-density
% of a multi-variate Gaussian at x.
%
% Input:
% x  - a nDim dimensional point (1-by-nDim vector).
% mu - mean (1-by-nDim vector).
% S  - covariance matrix (nDim-by-nDim matrix).
%
% Output:
% y  - log-density value (scalar).
%
% Date: August 6, 2018

    nDim = size(x, 2);
    const = -0.5 .* nDim .* log(2 .* pi);
    logDet = 2 .* sum(log(diag(chol(S))));
    xc = x - mu;
    y = const - 0.5 .* logDet - 0.5 .* xc / S * xc';
end
