%%
% File: genpts_fig1.m
% Purpose:
% Generate point-set plots for MCMC and SP-MCMC algorithms.
% Date: January 22, 2019
%%

% Add dependencies
addpath('../lib');

% Test case
testCase = 2;
if testCase == 1
    [Mu, C, w] = gmparam_2c(2, 1.5);
elseif testCase == 2
    [Mu, C, w] = gmparam_rnd();
end

% Number of points
nPts = 60;

% Target distribution
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Target dimension
nDim = size(Mu, 2);

% Optimiser
nIter = repmat(30, nPts, 1);
x0 = zeros(1, nDim);
S0 = 2 .* eye(nDim);
lambda = 1;
alpha = ones(nPts, 1);

% Symbolic variables
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Kernel
L = 3.2 .* eye(nDim);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) SP-Away
delete('fmin_grw_last_out.mat');
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);
[X1, ks1, evl1] = stein_greedy_away(nDim, fu, k, fmin, nPts, 1000);
n1 = cellfun(@(A)size(A, 1), X1);

% (2) MALA
nThin = 30;
nMala = n1(end) .* nThin;
[X, D, ~, acMala] = mala(fp, fu, x0, 1.5, nMala);
X2 = X(1:nThin:end, :);
D2 = D(1:nThin:end, :);
ks2 = ksd(X2, D2, k);
evl2 = repmat(2 .* nThin, n1(end), 1);

% KSD vs number of target evaluations
figure();
hold on;
plot(log(cumsum(evl1)), log(ks1), 'linewidth', 1);
plot(log(cumsum(evl2)), log(ks2), 'linewidth', 1);
xlabel('log n_{eval}');
ylabel('log KSD');
legend({'SP', 'MCMC'}, 'location', 'best');

% Growth SP-AWAY
figure();
plot(n1, 'linewidth', 1);
xlabel('Iteration');
ylabel('n');
title('Growth');

% Evaluate target density over a grid
lb = [-8, -8];
ub = [8, 8];
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = fp_gaussmix(T, Mu, C, w);
Z = reshape(p, nStep, nStep);

% Contour SP-Away
figure();
subplot(1, 2, 2);
contour(t1, t2, Z, ...
    'levelstep', 0.003, ...
    'linewidth', 1, ...
    'linecolor', [0.59, 0.59, 0.59]);
title('SP-MCMC', 'fontsize', 10);
axis equal;
hold on;
plot(X1{end}(:, 1), X1{end}(:, 2), '.r', 'markersize', 10);
set(gca, 'xtick', []);
set(gca, 'ytick', []);

% Contour MALA
subplot(1, 2, 1);
contour(t1, t2, Z, ...
    'levelstep', 0.003, ...
    'linewidth', 1, ...
    'linecolor', [0.59, 0.59, 0.59]);
title('MCMC', 'fontsize', 10);
axis equal;
hold on;
plot(X2(:, 1), X2(:, 2), '.r', 'markersize', 10);
set(gca, 'xtick', []);
set(gca, 'ytick', []);

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 12, 6]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 12, 6]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [12, 6]);

% Print to PDF
print('fig1', '-dpdf');
