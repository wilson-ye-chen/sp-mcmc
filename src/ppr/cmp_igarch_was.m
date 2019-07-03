%%
% File: cmp_igarch_was.m
% Purpose:
% Compare point-sets from various discrete approximation algorithms.
% The target is the posterior of an IGARCH model. The comparison is
% based on the Wasserstein distance.
% Date: July 2, 2019
%%

% Add dependencies
addpath('../lib');

% Drop sites
nPts = 300;
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

% Run MCMC to obtain EMDs and a preconditioner
nBurn = 1000;
nSamp = 100000;
x0 = (lb + ub) ./ 2;
Chain = mala(fp, fu, x0, 0.0035, eye(nDim), nBurn + nSamp);
Chain = Chain((nBurn + 1):end, :);

% Kernel
L = cov(Chain);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) MALA
nThin = 10;
nIter = nPts .* nThin;
[X, ~, ~, acMala] = mala(fp, fu, x0, 0.0035, eye(nDim), nIter);
X1 = X(1:nThin:end, :);
evl1 = repmat(2 .* nThin, nPts, 1);
Seg1 = [ones(nPts, 1), [1:nPts]'];

% (2) SVGD
nIter = 200;
X0 = [unifrnd(lb(1), ub(1), nPts, 1), unifrnd(lb(2), ub(2), nPts, 1)];
fstp = @(i, Phi)fstp_adagrad(i, Phi, 1e-3, 0.9);
[X, evl2] = mysvgd(fu, k, X0, fstp, nIter);
X2 = reshape(permute(X, [1, 3, 2]), [], 2, 1);
segEnd = nPts:nPts:(nPts .* nIter);
segBeg = segEnd - nPts + 1;
Seg2 = [segBeg', segEnd'];

% (3) MED
nMc = 10;
mu0 = (lb + ub) ./ 2;
S0 = 0.5 .* V;
lambda = 5e-6;
alpha = 1 - logistic(linspace(-1, 1, nPts), 8);
fmin = @(f, X)fmin_mc_med(f, X, nMc, mu0, S0, lambda, alpha);
[X3, ~, evl3] = med_greedy(nDim, fp, fmin, nPts);
Seg3 = [ones(nPts, 1), [1:nPts]'];

% (4) SP
nMc = 10;
mu0 = (lb + ub) ./ 2;
S0 = 0.5 .* V;
lambda = 5e-6;
alpha = 1 - logistic(linspace(-1, 1, nPts), 8);
fmin = @(f, X, G, fscr)fmin_mc( ...
    f, X, G, fscr, nMc, x0, S0, lambda, alpha);
[X, ~, evl4] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X4 = X{end};
Seg4 = [ones(nPts, 1), [1:nPts]'];

% (5) SP-MCMC (LAST)
nIter = repmat(10, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl5] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X5 = X{end};
Seg5 = [ones(nPts, 1), [1:nPts]'];

% (6) SP-MCMC (INFL)
nIter = repmat(10, nPts, 1);
x0 = (lb + ub) ./ 2;
S0 = 0.1 .* V;
lambda = 0.1;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl6] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X6 = X{end};
Seg6 = [ones(nPts, 1), [1:nPts]'];

% Evaluate log posterior (unnormalised) over a grid
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
Z = reshape(fp(T), nStep, nStep);

% Point-sets
lab = {'MCMC', 'SVGD', 'MED', 'SP', 'SP-MCMC LAST', 'SP-MCMC INFL'};
figure();
for i = 1:6
    subplot(2, 3, i);
    contour(t1, t2, Z, ...
        'levelstep', 2, ...
        'linewidth', 1, ...
        'linecolor', [0.6, 0.6, 0.6]);
    xlabel('\theta_1', 'fontsize', 13);
    ylabel('\theta_2', 'fontsize', 13);
    title(lab{i});
    hold on;
    X = eval(['X', num2str(i)]);
    m = size(X, 1);
    j = (m - nPts + 1):m;
    plot(X(j, 1), X(j, 2), '.r', 'markersize', 9);
end

% Trace plots
figure();
for i = 1:6
    subplot(2, 3, i);
    X = eval(['X', num2str(i)]);
    m = size(X, 1);
    j = (m - nPts + 1):m;
    plot(X(j, 1), 'linewidth', 1);
    axis([1, nPts, lb(1), ub(1)]);
    xlabel('Iteration', 'fontsize', 13);
    ylabel('\theta_1', 'fontsize', 13);
    title(lab{i});
end

% Compute Wasserstein distances
nIid = 1000;
nRep = 5;
Wass = cell(6, 1);
for i = 1:6
    YPts = eval(['X', num2str(i)]);
    SegY = eval(['Seg', num2str(i)]);
    m = size(SegY, 1);
    W = zeros(m, nRep);
    for j = 1:nRep
        XPts = Chain(randperm(nSamp, nIid), :);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia ../lib/l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        W(:, j) = emd;
    end
    Wass{i} = W;
end

% Wasserstein vs number of target evaluations
figure();
Rgb = get(gca, 'colororder');
set(gca, 'colororder', Rgb([5, 2, 3, 4, 6, 1], :));
hold on;
for i = 1:6
    evl = eval(['evl', num2str(i)]);
    plot(log(cumsum(evl)), log(mean(Wass{i}, 2)), 'linewidth', 1);
end
set(gca, 'fontsize', 16);
xlabel('log n_{eval}', 'fontsize', 16);
ylabel('log Wasserstein', 'fontsize', 16);
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

% Print to PDF
print('was_igarch', '-dpdf');

% Save output
save('was_igarch.mat');
