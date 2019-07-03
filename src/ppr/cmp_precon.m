%%
% File: cmp_precon.m
% Purpose:
% Compare point-sets generated with preconditioner kernel to those generated
% with median trick kernel. The target is the posterior of an IGARCH model.
% This script creates Figure S2 in the supplement of SP-MCMC.
% Date: July 2, 2019
%%

% Add dependencies
addpath('../lib');

% Drop sites
nPts = 1000;
drop = false(nPts, 1);

% Load S&P 500 return data
load('data_spx.mat');
str = 2501;
wdt = 2000;
r = r(str:(str + wdt - 1));

% GARCH score and log-density functions
h1 = var(r);
fu = @(X)fscr_igarch(X, r, h1);
fp = @(X)fp_igarch(X, r, h1);

% Basic info on target scale
lb = [0.002, 0.05];
ub = [0.04, 0.2];
V = diag([1e-4, 1e-3]);

% Symbolic variables
nDim = 2;
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Run MCMC for computing preconditioner and energy distances
nBurn = 1000;
nSamp = 100000;
nKeep = 20000;
x0 = (lb + ub) ./ 2;
Chain = mala(fp, fu, x0, 0.0035, eye(nDim), nBurn + nSamp);
Chain = Chain((nBurn + 1):end, :);
Chain = Chain(round(linspace(1, nSamp, nKeep)), :);

% Preconditioner kernel
L = cov(Chain);
LInv = inv(L);
k1 = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% Median trick kernel
ll = median(pdist(Chain)) .^ 2 ./ log(nPts);
L = ll .* eye(nDim);
LInv = inv(L);
k2 = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) SVGD-k1
nIter = 200;
X0 = [unifrnd(lb(1), ub(1), nPts, 1), unifrnd(lb(2), ub(2), nPts, 1)];
fstp = @(i, Phi)fstp_adagrad(i, Phi, 1e-3, 0.9);
[X, evl1] = mysvgd(fu, k1, X0, fstp, nIter);
X1 = X(:, :, end);
ed1 = zeros(nIter, 1);
for i = 1:nIter
    ed = mmd(X(:, :, i), Chain);
    ed1(i) = ed(end);
end

% (2) SP-k1
nMc = 5;
mu0 = (lb + ub) ./ 2;
S0 = 0.5 .* V;
lambda = 5e-6;
alpha = 1 - logistic(linspace(-0.5, 1.5, nPts), 20);
fmin = @(f, X, G, fscr)fmin_mc( ...
    f, X, G, fscr, nMc, x0, S0, lambda, alpha);
[X, ~, evl2] = stein_greedy_drop(nDim, fu, k1, fmin, drop);
X2 = X{end};
ed2 = mmd(X2, Chain);

% (3) INFL-k1
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl3] = stein_greedy_drop(nDim, fu, k1, fmin, drop);
X3 = X{end};
ed3 = mmd(X3, Chain);

% (4) SVGD-k2
nIter = 200;
X0 = [unifrnd(lb(1), ub(1), nPts, 1), unifrnd(lb(2), ub(2), nPts, 1)];
fstp = @(i, Phi)fstp_adagrad(i, Phi, 1e-3, 0.9);
[X, evl4] = mysvgd(fu, k2, X0, fstp, nIter);
X4 = X(:, :, end);
ed4 = zeros(nIter, 1);
for i = 1:nIter
    ed = mmd(X(:, :, i), Chain);
    ed4(i) = ed(end);
end

% (5) SP-k2
nMc = 5;
mu0 = (lb + ub) ./ 2;
S0 = 0.5 .* V;
lambda = 5e-6;
alpha = 1 - logistic(linspace(-0.5, 1.5, nPts), 20);
fmin = @(f, X, G, fscr)fmin_mc( ...
    f, X, G, fscr, nMc, x0, S0, lambda, alpha);
[X, ~, evl5] = stein_greedy_drop(nDim, fu, k2, fmin, drop);
X5 = X{end};
ed5 = mmd(X5, Chain);

% (6) INFL-k2
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl6] = stein_greedy_drop(nDim, fu, k2, fmin, drop);
X6 = X{end};
ed6 = mmd(X6, Chain);

% Energy distances vs number of target evaluations
lab = {'SVGD Pre.', 'SP Pre.', 'SP-MCMC Pre.', 'SVGD', 'SP', 'SP-MCMC'};
figure();
hold on;
for i = 1:6
    evl = eval(['evl', num2str(i)]);
    ed = eval(['ed', num2str(i)]);
    plot(log(cumsum(evl)), log(ed), 'linewidth', 1);
end
set(gca, 'fontsize', 16);
xlabel('log n_{eval}', 'fontsize', 16);
ylabel('log E_P', 'fontsize', 16);
legend(lab, 'location', 'best', 'fontsize', 13);

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 20, 12]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 20, 12]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [20, 12]);

% Print energy distance plot to PDF
print('ed_precon', '-dpdf');

% Save output
save('ed_precon.mat');
