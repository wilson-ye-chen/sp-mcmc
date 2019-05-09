%%
% File: cmp_igarch.m
% Purpose:
% Compare point-sets from various discrete approximation algorithms.
% The target is the posterior of an IGARCH model. This script creates
% Figure 3 in SP-MCMC.
% Date: May 9, 2019
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
V = eye(2) .* 5e-4;

% Symbolic variables
nDim = 2;
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Run MCMC for computing preconditioner and energy distances
nSamp = 100000;
nKeep = 20000;
x0 = (lb + ub) ./ 2;
epoch = [repmat(500, 15, 1); 1000; 5000; nSamp];
alpha = 1 - logistic(linspace(-0.1, 1.9, numel(epoch)), 20);
[hOptm, ~, Chn] = mala_adapt(fp, fu, x0, 0.004, eye(nDim), alpha, epoch);
Chain = Chn{end};
Chain = Chain(round(linspace(1, nSamp, nKeep)), :);

% Kernel
L = cov(Chain);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) MALA
nThin = 5;
nIter = nPts .* nThin;
x0 = (lb + ub) ./ 2;
[X, D, ~, acMala] = mala(fp, fu, x0, 0.13, V, nIter);
X1 = X(1:nThin:end, :);
D1 = D(1:nThin:end, :);
ks1 = ksd(X1, D1, k);
ed1 = mmd(X1, Chain);
evl1 = repmat(2 .* nThin, nPts, 1);

% (2) GRW
nThin = 5;
nIter = nPts .* nThin;
x0 = (lb + ub) ./ 2;
[X, acRw] = grwmetrop(fp, x0, 0.06 .* V, nIter);
X2 = X(1:nThin:end, :);
ks2 = ksd(X2, fu(X2), k);
ed2 = mmd(X2, Chain);
evl2 = repmat(nThin, nPts, 1);

% (3) SVGD
nIter = 200;
X0 = [unifrnd(lb(1), ub(1), nPts, 1), unifrnd(lb(2), ub(2), nPts, 1)];
fstp = @(i, Phi)fstp_adagrad(i, Phi, 1e-3, 0.9);
[X, evl3] = mysvgd(fu, k, X0, fstp, nIter);
X3 = X(:, :, end);
ks3 = zeros(nIter, 1);
ed3 = zeros(nIter, 1);
for i = 1:nIter
    ks = ksd(X(:, :, i), fu(X(:, :, i)), k);
    ed = mmd(X(:, :, i), Chain);
    ks3(i) = ks(end);
    ed3(i) = ed(end);
end

% (4) MED
nMc = 5;
mu0 = (lb + ub) ./ 2;
S0 = 0.5 .* V;
lambda = 5e-6;
alpha = 1 - logistic(linspace(-0.15, 1.85, nPts), 50);
alpha(1) = 1;
fmin = @(f, X)fmin_mc_med(f, X, nMc, mu0, S0, lambda, alpha);
[X4, ~, evl4] = med_greedy(nDim, fp, fmin, nPts);
ks4 = ksd(X4, fu(X4), k);
ed4 = mmd(X4, Chain);

% (5) SP
nMc = 5;
mu0 = (lb + ub) ./ 2;
S0 = 0.5 .* V;
lambda = 5e-6;
alpha = 1 - logistic(linspace(-0.15, 1.85, nPts), 50);
alpha(1) = 1;
fmin = @(f, X, G, fscr)fmin_mc( ...
    f, X, G, fscr, nMc, x0, S0, lambda, alpha);
[X, ks5, evl5] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X5 = X{end};
ed5 = mmd(X5, Chain);

% (6) SP-MALA (LAST)
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
alpha = 1 - logistic(linspace(-0.1, 1.9, nPts), 100);
alpha(1:10) = 1;
fmin = @(f, X, G, fscr)fmin_mala_last( ...
    f, X, G, fscr, fp, nIter, x0, V ./ 0.5 .^ 2 .* 0.02, 0.5, alpha, false);
[X, ks6, evl6] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X6 = X{end};
ed6 = mmd(X6, Chain);

% (7) SP-MALA (INFL)
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
alpha = 1 - logistic(linspace(-0.1, 1.9, nPts), 100);
alpha(1:10) = 1;
fmin = @(f, X, G, fscr)fmin_mala_infl( ...
    f, X, G, fscr, fp, nIter, x0, V ./ 0.8 .^ 2 .* 0.02, 0.8, alpha, false);
[X, ks7, evl7] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X7 = X{end};
ed7 = mmd(X7, Chain);

% (8) SP-GRW (LAST)
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
lambda = 2.38 .^ 2 ./ nDim;
alpha = 1 - logistic(linspace(-0.1, 1.9, nPts), 100);
alpha(1:10) = 1;
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fp, nIter, x0, 0.2 .* V, lambda, alpha, false);
[X, ks8, evl8] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X8 = X{end};
ed8 = mmd(X8, Chain);

% (9) SP-GRW (INFL)
nIter = repmat(5, nPts, 1);
x0 = (lb + ub) ./ 2;
lambda = 2.38 .^ 2 ./ nDim;
alpha = 1 - logistic(linspace(-0.1, 1.9, nPts), 100);
alpha(1:10) = 1;
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, 0.2 .* V, lambda, alpha, false);
[X, ks9, evl9] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X9 = X{end};
ed9 = mmd(X9, Chain);

% Evaluate log posterior (unnormalised) over a grid
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
Z = reshape(fp(T), nStep, nStep);

% Plot appearance
lab = {
    'MALA', ...
    'RWM', ...
    'SVGD', ...
    'MED', ...
    'SP', ...
    'SP-MALA LAST', ...
    'SP-MALA INFL', ...
    'SP-RWM LAST', ...
    'SP-RWM INFL'};
Rgb = [
    0.000, 0.447  0.741; ...
    0.494, 0.184, 0.556; ...
    0.850, 0.325, 0.098; ...
    0.929, 0.694, 0.125; ...
    0.466, 0.674, 0.188; ...
    0.000, 0.447  0.741; ...
    0.301, 0.745, 0.933; ...
    0.494, 0.184, 0.556; ...
    0.933, 0.510, 0.933];
lnSty = {':', ':', '-', '-', '-', '-', '-', '-', '-'};
lnWdt = [1.7, 1.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

% Point-sets
figure();
for i = 1:9
    subplot(3, 3, i);
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
set(gcf, 'position', [0.5, 1.5, 20, 20]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 20, 20]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [20, 20]);

% Print point-set plots to PDF
print('pts_igarch', '-dpdf');

% KSD vs number of target evaluations
figure();
set(gca, 'colororder', Rgb);
hold on;
for i = 1:9
    evl = eval(['evl', num2str(i)]);
    ks = eval(['ks', num2str(i)]);
    plot(log(cumsum(evl)), log(ks), lnSty{i}, 'linewidth', lnWdt(i));
end
set(gca, 'fontsize', 16);
xlabel('log n_{eval}', 'fontsize', 16);
ylabel('log KSD', 'fontsize', 16);
legend(lab, 'location', 'best', 'fontsize', 12);

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

% Print KSD plot to PDF
print('ksd_igarch', '-dpdf');

% Energy distances vs number of target evaluations
figure();
set(gca, 'colororder', Rgb);
hold on;
for i = 1:9
    evl = eval(['evl', num2str(i)]);
    ed = eval(['ed', num2str(i)]);
    plot(log(cumsum(evl)), log(ed), lnSty{i}, 'linewidth', lnWdt(i));
end
set(gca, 'fontsize', 16);
xlabel('log n_{eval}', 'fontsize', 16);
ylabel('log E_P', 'fontsize', 16);
axis([1.6, 12.3, -11.8, -3.4]);
legend(lab, 'location', 'best', 'fontsize', 12);

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
print('ed_igarch', '-dpdf');

% Save output
save('ksd_ed_igarch.mat');
