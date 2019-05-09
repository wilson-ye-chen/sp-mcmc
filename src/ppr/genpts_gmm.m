%%
% File: genpts_gmm.m
% Purpose:
% Generates point-set plots for various discrete approximation algorithms.
% The target is a two-dimensional standard Gaussian mixture model.
% Date: January 31, 2019
%%

% Add dependencies
addpath('../lib');

% Number of particles
nPart = 1000;

% Target density and score functions
[Mu, C, w] = gmparam_2c(2, 0.5);
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% MCMC optimiser setup
nDim = size(Mu, 2);
nIter = repmat(5, nPart, 1);
x0 = zeros(1, nDim);
S0 = diag(0.1 .* ones(1, nDim));
lambda = 0.1;
alpha = ones(nPart, 1);

% Symbolic variables
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Kernel
X = gaussmix_rnd(Mu, C, w, nPart);
ll = median(pdst(X)) .^ 2 ./ log(nPart);
L = ll .* eye(nDim);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) MALA
[xArr{1}, ~, ~, acMala] = mala(fp, fu, x0, 1.1, eye(nDim), nPart);

% (2) LAST
delete('fmin_grw_last_out.mat');
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
xArr{2} = stein_greedy(nDim, fu, k, fmin, nPart);

% (3) RAND
delete('fmin_grw_rand_out.mat');
fmin = @(f, X, G, fscr)fmin_grw_rand( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
xArr{3} = stein_greedy(nDim, fu, k, fmin, nPart);

% (4) INFL
delete('fmin_grw_infl_out.mat');
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
xArr{4} = stein_greedy(nDim, fu, k, fmin, nPart);

% Save point sets
save('genpts_gmm_out.mat', ...
    'nDim', 'Mu', 'C', 'w', 'fp', 'fu', ...
    'a', 'b', 'LInv', 'k', ...
    'nPart', 'xArr');
