function [h, C, X, a] = grwmetrop_adapt(fp, x0, h0, C0, alpha, epoch)
% [h, C, X, a] = grwmetrop_adapt(fp, x0, h0, C0, alpha, epoch) samples
% from a target distribution using an adaptive version of the random-walk
% Metropolis algorithm.
%
% Input:
% fp    - handle to the log-density function of the target.
% x0    - vector of the starting values of the Markov chain.
% h0    - initial step-size parameter.
% C0    - initial preconditioning matrix.
% alpha - adaptive schedule.
% epoch - length of each tuning epoch.
%
% Output:
% h     - tuned step-size.
% C     - tuned preconditioning matrix.
% X     - cell array of matrices of generated points.
% a     - cell array of binary vectors indicating whether a move is accepted.
%
% Date: April 5, 2019

    nEp = numel(epoch);
    X = cell(nEp, 1);
    a = cell(nEp, 1);

    % First epoch
    h = h0;
    [X{1}, a{1}] = grwmetrop(fp, x0, h .^ 2 .* C0, epoch(1));
    fprintf('Epoch = 1\n');

    for i = 2:nEp
        % Adapt preconditioning matrix
        if alpha(i) == 1
            C = C0;
        else
            C = alpha(i) .* C0 + (1 - alpha(i)) .* cov(X{i - 1});
            C = nearestSPD(C);
        end

        % Tune step-size
        ar = sum(a{i - 1}) ./ epoch(i - 1);
        h = h .* exp(ar - 0.234);

        % Next epoch
        x0 = X{i - 1}(end, :);
        [X{i}, a{i}] = grwmetrop(fp, x0, h .^ 2 .* C, epoch(i));
        fprintf('Epoch = %d\n', i);
    end
end
