function [xMin, dMin, k0Min, nEval] = fmin_mc( ...
    f, X, G, fscr, nMc, mu0, S0, lambda, alpha)
% [xMin, dMin, fMin, nEval] = fmin_adamc(f, X, G, fscr, nMc, mu0, S0, ...
% lambda, alpha) minimises the objective function using an adaptive Monte
% Carlo optimisation approach where the proposal distributions is created
% adaptively using a mixture of Gaussians based on the current points.
%
% Date: January 18, 2019

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
    D = fscr(XMc);
    [fVal, K0] = f(XMc, D);
    [~, iMin] = min(fVal);
    xMin = XMc(iMin, :);
    dMin = D(iMin, :);
    k0Min = K0(iMin, :);
    nEval = nMc;
end
