%%
% File: cmp_away.m
% Purpose:
% Compare point-sets from various discrete approximation algorithms.
% The target is the posterior of an IGARCH model. This script creates
% Figure S3 in the supplement of SP-MCMC.
% Date: July 2, 2019
%%

% Add dependencies
addpath('../lib');

% Drop sites
nPts = 1000;

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

% Kernel
L = cov(Chain);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) INFL
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
drop = false(nPts, 1);
[X, ~, evl1] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X1 = X{end};
ed1 = mmd(X1, Chain);

% (2) INFL Away
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl2] = stein_greedy_away(nDim, fu, k, fmin, nPts, nPts);
X2 = X{end};
ed2 = zeros(nPts, 1);
for i = 1:nPts
    ed = mmd(X{i}, Chain);
    ed2(i) = ed(end);
end
n2 = cellfun(@(A)size(A, 1), X);

% (3) INFL Drop
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
drop = false(nPts + 300, 1);
drop(round(linspace(10, nPts + 300, 300))) = true;
[X, ~, evl3] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X3 = X{end};
ed3 = zeros(nPts, 1);
for i = 1:numel(drop)
    ed = mmd(X{i}, Chain);
    ed3(i) = ed(end);
end
n3 = cellfun(@(A)size(A, 1), X);

% Evaluate log posterior (unnormalised) over a grid
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
Z = reshape(fp(T), nStep, nStep);

% Point-sets
lab = {'SP-MCMC INFL', 'SP-MCMC INFL Away', 'SP-MCMC INFL Drop'};
figure();
for i = 1:3
    subplot(1, 3, i);
    contour(t1, t2, Z, ...
        'levelstep', 2, ...
        'linewidth', 1, ...
        'linecolor', [0.6, 0.6, 0.6]);
    xlabel('\theta_1', 'fontsize', 13);
    ylabel('\theta_2', 'fontsize', 13);
    title(lab{i});
    hold on;
    X = eval(['X', num2str(i)]);
    plot(X(:, 1), X(:, 2), '.r', 'markersize', 8);
    axis([lb(1), ub(1), lb(2), ub(2)]);
    box on;
end

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 25, 7]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 25, 7]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [25, 7]);

% Print point-set plots to PDF
print('pts_away', '-dpdf');

% Energy distances vs number of target evaluations
figure();
hold on;
for i = 1:3
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
print('ed_away', '-dpdf');

% Save output
save('ed_away.mat');
