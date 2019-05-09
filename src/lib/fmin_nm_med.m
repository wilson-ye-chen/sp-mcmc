function [xMin, fMin, nEval] = fmin_nm_med(f, X, nIni, mu0, S0, lambda, alpha)
% [xMin, fMin, nEval] = fmin_nm_med(f, X, nIni, mu0, S0, lambda, alpha)
% minimises the objective function using a Nelder-Mead algorithm, where a
% Nelder-Mead optimiser is restarted multiple times, with adaptively chosen
% initial values based on the current point-set. This particular optimiser is
% not compatible with Stein greedy algorithms.
%
% Date: January 21, 2019

    [nObs, nDim] = size(X);
    iNew = nObs + 1;

    % Adapt proposal based on X
    if rand() <= alpha(iNew)
        frnd = @()mvnrnd(mu0, S0, 1);
    else
        S = lambda .* eye(nDim);
        frnd = @()mvnrnd(X(randi(nObs), :), S, 1);
    end

    % Nelder–Mead method for multiple local searches
    opt = optimset('tolfun', 1e-3, 'tolx', 1e-3, 'display', 'off');
    XLoc = zeros(nIni, nDim);
    fLoc = zeros(nIni, 1);
    nEval = 0;
    for i = 1:nIni
        [XLoc(i, :), fLoc(i), ~, out] = fminsearch(f, frnd(), opt);
        nEval = nEval + out.funcCount;
    end

    % Obtain 'global' solution
    [fMin, iMin] = min(fLoc);
    xMin = XLoc(iMin, :);
end
