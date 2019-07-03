%%
% File: test_gaussmix_replay.m
% Purpose:
% Creates a video output from SP-MCMC for a GMM target.
% Date: July 2, 2019
%%

% Add dependencies
addpath('../lib');

% Number of particles
nPart = 1000;

% Target density and score functions
[Mu, C, w] = gmparam_2c(2, 0.8);
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

% SP-RWM INFL
% Minimiser output file is needed for the creating video output
delete('fmin_grw_infl_out.mat');
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
X = stein_greedy(nDim, fu, k, fmin, nPart);

% Generate video output
load('fmin_grw_infl_out.mat');
d = [1, 2];
lb = [-3.5, -3.5];
ub = [3.5, 3.5];
nStep = 100;
out = 'replay_grw_infl.avi';
[h1, h2] = gaussmix_replay(X, SttPt, Mu, C, w, d, lb, ub, nStep, out);
