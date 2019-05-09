%%
% File: genpts_igarch.m
% Purpose:
% Generates point-set plots for various discrete approximation algorithms.
% The target is the posterior of an IGARCH model.
% Date: February 7, 2019
%%

% Add dependencies
addpath('../lib');

% Drop sites
nPart = 1000;
drop = false(nPart, 1);

% Load S&P 500 return data
load('data_spx.mat');
str = 2501;
wdt = 2000;
r = r(str:(str + wdt - 1));

% GARCH score and log-density functions
v1 = var(r);
fu = @(X)fscr_igarch(X, r, v1);
fp = @(X)fp_igarch(X, r, v1);

% MCMC optimiser setup
nIter = repmat(5, nPart, 1);
lb = [0.002, 0.05];
ub = [0.04, 0.2];
x0 = (lb + ub) ./ 2;

% Symbolic variables
nDim = 2;
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Run MCMC to obtain a preconditioner
epoch = [repmat(100, 50, 1); 500; 1000; 5000];
alpha = 1 - logistic(linspace(-0.5, 1.5, numel(epoch)), 10);
[hOptm, ~, Chn] = mala_adapt(fp, fu, x0, 0.004, eye(nDim), alpha, epoch);
Chain = Chn{end};

% Kernel
L = cov(Chain);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) MALA
nThin = 5;
nMala = nPart .* nThin;
[X, D, ~, acMala] = mala(fp, fu, x0, hOptm, L, nMala);
X1 = X(1:nThin:end, :);
D1 = D(1:nThin:end, :);
ks1 = ksd(X1, D1, k);
evl1 = repmat(2 .* nThin, nPart, 1);

% (2) LAST
h = 0.8;
alpha = ones(nPart, 1);
fmin = @(f, X, G, fscr)fmin_mala_last( ...
    f, X, G, fscr, fp, nIter, x0, L, h, alpha, false);
[X, ks2, evl2] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X2 = X{end};

% (3) RAND
h = 0.8;
alpha = ones(nPart, 1);
fmin = @(f, X, G, fscr)fmin_mala_rand( ...
    f, X, G, fscr, fp, nIter, x0, L, h, alpha, false);
[X, ks3, evl3] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X3 = X{end};

% (4) INFL
h = 0.8;
alpha = ones(nPart, 1);
fmin = @(f, X, G, fscr)fmin_mala_infl( ...
    f, X, G, fscr, fp, nIter, x0, L, h, alpha, false);
[X, ks4, evl4] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X4 = X{end};

% Evaluate log posterior (unnormalised) over a grid
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
Z = reshape(fp(T), nStep, nStep);

% Contour 'MCMC'
figure();
subplot(2, 2, 1);
contour(t1, t2, Z, ...
    'levelstep', 2, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('\theta_1', 'fontsize', 13);
ylabel('\theta_2', 'fontsize', 13);
title('MCMC');
hold on;
plot(X1(:, 1), X1(:, 2), '.r', 'markersize', 8);

% Contour 'LAST'
subplot(2, 2, 2);
contour(t1, t2, Z, ...
    'levelstep', 2, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('\theta_1', 'fontsize', 13);
ylabel('\theta_2', 'fontsize', 13);
title('LAST');
hold on;
plot(X2(:, 1), X2(:, 2), '.r', 'markersize', 8);

% Contour 'RAND'
subplot(2, 2, 3);
contour(t1, t2, Z, ...
    'levelstep', 2, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('\theta_1', 'fontsize', 13);
ylabel('\theta_2', 'fontsize', 13);
title('RAND');
hold on;
plot(X3(:, 1), X3(:, 2), '.r', 'markersize', 8);

% Contour 'INFL'
subplot(2, 2, 4);
contour(t1, t2, Z, ...
    'levelstep', 2, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('\theta_1', 'fontsize', 13);
ylabel('\theta_2', 'fontsize', 13);
title('INFL');
hold on;
plot(X4(:, 1), X4(:, 2), '.r', 'markersize', 8);

% Trace 'MCMC'
figure();
subplot(2, 2, 1);
plot(X1(:, 1), 'linewidth', 1);
axis([1, 1000, lb(1), ub(1)]);
xlabel('Iteration', 'fontsize', 13);
ylabel('\theta_1', 'fontsize', 13);
title('MCMC');

% Trace 'LAST'
subplot(2, 2, 2);
plot(X2(:, 1), 'linewidth', 1);
axis([1, 1000, lb(1), ub(1)]);
xlabel('Iteration', 'fontsize', 13);
ylabel('\theta_1', 'fontsize', 13);
title('LAST');

% Trace 'RAND'
subplot(2, 2, 3);
plot(X3(:, 1), 'linewidth', 1);
axis([1, 1000, lb(1), ub(1)]);
xlabel('Iteration', 'fontsize', 13);
ylabel('\theta_1', 'fontsize', 13);
title('RAND');

% Trace 'INFL'
subplot(2, 2, 4);
plot(X4(:, 1), 'linewidth', 1);
axis([1, 1000, lb(1), ub(1)]);
xlabel('Iteration', 'fontsize', 13);
ylabel('\theta_1', 'fontsize', 13);
title('INFL');

% KSD vs number of target evaluations
figure();
hold on;
plot(log(cumsum(evl1)), log(ks1), 'linewidth', 1);
plot(log(cumsum(evl2)), log(ks2), 'linewidth', 1);
plot(log(cumsum(evl3)), log(ks3), 'linewidth', 1);
plot(log(cumsum(evl4)), log(ks4), 'linewidth', 1);
set(gca, 'fontsize', 13);
xlabel('log n_{eval}', 'fontsize', 13);
ylabel('log KSD', 'fontsize', 13);
legend({'MCMC', 'LAST', 'RAND', 'INFL'}, ...
    'location', 'best', ...
    'fontsize', 13);
