function [X, D, p, a] = mala(fp, fscr, x0, h, C, nIter)
% [X, D, p, a] = mala(fp, fscr, x0, h, C, nIter) samples from
% a target distribution using the Metropolis-adjusted Langevin
% algorithm.
%
% Input:
% fp    - handle to the log-density function of the target.
% fscr  - handle to the gradient function of the log target.
% x0    - vector of the starting values of the Markov chain.
% h     - step-size parameter.
% C     - preconditioning matrix.
% nIter - number of MCMC iterations.
%
% Output:
% X     - matrix of generated points.
% D     - matrix of gradients of the log target at X.
% p     - vector of log-density values of the target at X.
% a     - binary vector indicating whether a move is accepted.
%
% Date: February 4, 2019

    % Initialise the chain
    nDim = numel(x0);
    X = zeros(nIter, nDim);
    D = zeros(nIter, nDim);
    p = zeros(nIter, 1);
    a = zeros(nIter, 1);
    X(1, :) = x0;
    D(1, :) = fscr(x0);
    p(1) = fp(x0);

    % For each MCMC iteration
    for i = 2:nIter
        % Langevin proposal
        hh = h .^ 2;
        mx = X(i - 1, :) + hh ./ 2 .* D(i - 1, :) * C;
        S = hh .* C;
        y = mvnrnd(mx, S);

        % Log acceptance probability
        py = fp(y);
        dy = fscr(y);
        my = y + hh ./ 2 .* dy * C;
        qx = logmvnpdf(X(i - 1, :), my, S);
        qy = logmvnpdf(y, mx, S);
        accPr = (py + qx) - (p(i - 1) + qy);

        % Accept with probability accPr
        if accPr >= 0 || log(unifrnd(0, 1)) < accPr
            X(i, :) = y;
            D(i, :) = dy;
            p(i) = py;
            a(i) = 1;
        else
            X(i, :) = X(i - 1, :);
            D(i, :) = D(i - 1, :);
            p(i) = p(i - 1);
        end
    end
end
