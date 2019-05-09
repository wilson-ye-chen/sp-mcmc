function [d2X, d2Max, d2Sel, d2Lst] = jumpsize(X, Chain, obIdx)
% [d2X, d2Max, d2Sel, d2Lst] = jumpsize(X, Chain, obIdx) computes the jump
% sizes of the Markov chain and those of the Stein points. This function is
% unable to correctly handle dropping.
%
% Date: December 13, 2018

    % Squared jump sizes of Stein points
    d2X = sum(diff(X, 1, 1) .^ 2, 2);

    % Jump sizes of the Markov chains
    nChn = size(X, 1);
    d2Max = zeros(nChn, 1);
    d2Sel = zeros(nChn, 1);
    d2Lst = zeros(nChn, 1);
    for i = 1:nChn
        C = Chain(obIdx == i, :);
        d2 = sum((repmat(C(1, :), size(C, 1), 1) - C) .^ 2, 2);
        d2Max(i) = max(d2);
        d2Sel(i) = sum((C(1, :) - X(i, :)) .^ 2, 2);
        d2Lst(i) = d2(end);
    end
end
