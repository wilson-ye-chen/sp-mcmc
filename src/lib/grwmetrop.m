function [X, a] = grwmetrop(fp, x0, S, nIter)
% [X, a] = grwmetrop(fp, x0, S, nIter) samples from a target distribution
% using the random-walk Metropolis algorithm with a Gaussian proposal.
%
% Input:
% fp    - handle to the log kernel function of the target density.
% x0    - vector of the starting values of the Markov chain.
% S     - covariance matrix of the Gaussian proposal distribution.
% nIter - number of MCMC iterations.
%
% Output:
% X     - Markov chain of points.
% a     - vector of indicators for whether a move is accepted.
%
% Date: July 26, 2018

    % Initialise the chain
    nDim = numel(x0);
    X = zeros(nIter, nDim);
    X(1, :) = x0;

    % Evaluate log-kernel of starting point
    pOld = fp(x0);

    % Acceptance indicators
    a = zeros(nIter, 1);

    % For each MCMC iteration
    for i = 2:nIter
        % Copy forward the chain
        X(i, :) = X((i - 1), :);

        % Propose from a Gaussian
        pps = mvnrnd(X(i - 1, :), S, 1);

        % Compute the log acceptance probability
        p = fp(pps);
        accPr = p - pOld;

        % Accept with probability accPr
        if accPr >= 0 || log(unifrnd(0, 1)) < accPr
            X(i, :) = pps;
            pOld = p;
            a(i) = 1;
        end
    end
end
