function [X, negOb, nEval] = med_greedy(nDim, flogp, fmin, nIter)
% [X, negOb, nEval] = med_greedy(nDim, flogp, fmin, nIter) generates a point-
% set called minimum energy design (MED) using a one-point-at-a-time greedy
% algorithm described by Joseph et al (2017).
%
% Input:
% nDim  - number of dimensions of the target density.
% flogp - handle to the log-density function of the target.
% fmin  - handle to a nDim-dimensional minimiser.
% nIter - length of the generated sequence of points.
%
% Output:
% X     - nIter-by-nDim matrix of generated points.
% negOb - value of the negative MED objective at each iteration.
% nEval - number of density evaluations at each iteration.
%
% Date: January 17, 2019

    X = zeros(nIter, nDim);
    lp = zeros(nIter, 1);
    negOb = zeros(nIter, 1);
    nEval = zeros(nIter, 1);

    % Generate x_1
    f = @(XNew)-flogp(XNew);
    [X(1, :), negOb(1), nEval(1)] = fmin(f, double.empty(0, nDim));
    lp(1) = -negOb(1);
    fprintf('n = 1\n');

    % Generate the rest
    for n = 2:nIter
        f = @(XNew)-medobj(XNew, flogp, X, lp, n);
        [X(n, :), negOb(n), nEval(n)] = fmin(f, X(1:(n - 1), :));
        lp(n) = flogp(X(n, :));
        nEval(n) = nEval(n) + 1;
        fprintf('n = %d\n', n);
    end
end

function ob = medobj(XNew, flogp, X, lp, n)
    [nNew, nDim] = size(XNew);
    A = repmat(XNew, n - 1, 1);
    B = repelem(X(1:(n - 1), :), nNew, 1);
    D = reshape(logdst(A, B), nNew, []);
    P = repmat(lp(1:(n - 1))', nNew, 1);
    PNew = repmat(flogp(XNew), 1, n - 1);
    ob = min(PNew + P + 2 .* nDim .* D, [], 2);
end

function ld = logdst(A, B)
    ld = 0.5 .* log(sum((A - B) .^ 2, 2));
end
