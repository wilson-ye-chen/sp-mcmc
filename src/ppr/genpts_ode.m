%%
% File: genpts_ode.m
% Purpose:
% Generates point-set plots for various discrete approximation algorithms.
% The target is the posterior of a Goodwin oscillator.
% Date: April 6, 2019
%%

% Add dependencies
addpath('../lib');

% Test cases
testCase = 2;
if testCase == 1
    g = 2;
    lenScl = 0.3;
    chnLen = 10;
    prpScl = 0.003;
    malStp = 0.04;
elseif testCase == 2
    g = 8;
    lenScl = 0.15;
    chnLen = 20;
    prpScl = 0.0002;
    malStp = 0.007;
    preCon = 'precon_ode_10d.mat';
end

% Number of particles
nPts = 300;
drop = false(nPts, 1);

% Target density and score functions
fp = @(X)goodwin(X, g);
fu = @(X)out2(@()goodwin(X, g));

% Symbolic variables
nDim = g + 2;
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Kernel
L = lenScl .* eye(nDim);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% MCMC optimiser setup
nIter = repmat(chnLen, nPts, 1);
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);

% Run adaptive Metropolis to re-generate preconditioner
% h0 = 2.38 ./ sqrt(nDim);
% epoch = [repmat(500, 20, 1); repmat(1000, 10, 1); 5000; 10000; 20000];
% alpha = 1 - logistic(linspace(-0.1, 1.9, numel(epoch)), 20);
% [hRw, ~, XRw, acRw] = grwmetrop_adapt(fp, x0, h0, eye(nDim), alpha, epoch);
% V = cov(XRw{end});
% save(preCon, 'V');

% Load preconditioner
load(preCon);

% (1) MALA
nThin = chnLen;
nMala = nPts .* nThin;
[X, D, ~, acMala] = mala(fp, fu, x0, 0.07, V, nMala);
X1 = X(1:nThin:end, :);
D1 = D(1:nThin:end, :);
ks1 = ksd(X1, D1, k);
evl1 = repmat(2 .* nThin, nPts, 1);

% (2) GRW-LAST
delete('fmin_grw_last_out.mat');
S0 = 2.38 .^ 2 ./ nDim .* V;
lambda = 2.38 .^ 2 ./ nDim;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
[X2, ks2, evl2] = stein_greedy_drop(nDim, fu, k, fmin, drop);

% (3) GRW-INFL
delete('fmin_grw_infl_out.mat');
S0 = 2.38 .^ 2 ./ nDim .* V;
lambda = 2.38 .^ 2 ./ nDim;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
[X3, ks3, evl3] = stein_greedy_drop(nDim, fu, k, fmin, drop);

% (4) MALA-LAST
delete('fmin_mala_last_out.mat');
h = 0.07;
S0 = 0.001 .* eye(nDim);
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_mala_last( ...
    f, X, G, fscr, fp, nIter, x0, V, h, alpha, true);
[X4, ks4, evl4] = stein_greedy_drop(nDim, fu, k, fmin, drop);

% (5) MALA-INFL
delete('fmin_mala_infl_out.mat');
h = 0.07;
S0 = 0.001 .* eye(nDim);
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_mala_infl( ...
    f, X, G, fscr, fp, nIter, x0, V, h, alpha, true);
[X5, ks5, evl5] = stein_greedy_drop(nDim, fu, k, fmin, drop);

% Generate KSD plot
figure();
hold on;
plot(log(cumsum(evl1)), log(ks1), 'linewidth', 1);
plot(log(cumsum(evl2)), log(ks2), 'linewidth', 1);
plot(log(cumsum(evl3)), log(ks3), 'linewidth', 1);
plot(log(cumsum(evl4)), log(ks4), 'linewidth', 1);
plot(log(cumsum(evl5)), log(ks5), 'linewidth', 1);
set(gca, 'fontsize', 13);
xlabel('log n_{eval}', 'fontsize', 13);
ylabel('log KSD', 'fontsize', 13);
legend({'MCMC', 'GRW-LAST', 'GRW-INFL', 'MALA-LAST', 'MALA-INFL'}, ...
    'location', 'best', 'fontsize', 13);
