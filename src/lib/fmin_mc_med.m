function [xMin, fMin, nEval] = fmin_mc_med(f, X, nMc, mu0, S0, lambda, alpha)
% [xMin, fMin, nEval] = fmin_mc_med(f, X, nMc, mu0, S0, lambda, alpha)
% minimises the objective function using an adaptive Monte Carlo optimisation
% approach where the proposal distributions is created adaptively using a
% mixture of Gaussians based on the current points. This particular optimiser
% is not compatible with Stein greedy algorithms.
%
% Date: January 21, 2019

    [nObs, nDim] = size(X);
    iNew = nObs + 1;

    % Adapt proposal based on X
    if rand() <= alpha(iNew)
        XMc = mvnrnd(mu0, S0, nMc);
    else
        i = randi(nObs, nMc, 1);
        S = lambda .* eye(nDim);
        XMc = mvnrnd(X(i, :), S, nMc);
    end

    % Evaluate vectorised objective
    [fMin, iMin] = min(f(XMc));
    xMin = XMc(iMin, :);
    nEval = nMc;
end
