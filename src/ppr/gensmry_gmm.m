%%
% File: gensmry_gmm.m
% Purpose:
% Generates summary plots for various discrete approximation algorithms.
% The target is a two-dimensional standard Gaussian mixture model.
% Date: January 12, 2019
%%

% Add dependencies
addpath('../lib');

% Load point sets
load('genpts_gmm_out.mat');

% Kernel
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% Contour calculations
d = [1, 2];
lb = [-3.5, -3.5];
ub = [3.5, 3.5];
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = fp_gaussmix(T, Mu(:, d), C(d, d, :), w);
Z = reshape(p, nStep, nStep);

% Plot appearance
axTrc = [1, nPart, -3.5, 3.5];
axKsd = [1, nPart, -4.2, 1];
axJmp = [-0.2, 6, 0, 1.5];
lvStp = 0.01;
lnWdt = 1;
lnRgb = [0.7, 0.7, 0.7];
ptSiz = 5;
tcSiz = 4;
mcRgb = [255, 127, 14] ./ 255;
spRgb = [44, 160, 44] ./ 255;
ttFnt = 16;
tkFnt = 16;
lbFnt = 16;
lgFnt = 10;
ptTrn = 5;
tcTrn = 10;

% (1) MALA
X = xArr{1};
subplot(4, 4, 1);
contour(t1, t2, Z, ...
    'levelstep', lvStp, ...
    'linewidth', lnWdt, ...
    'linecolor', lnRgb);
d1 = d(1);
d2 = d(2);
title('MCMC', 'fontsize', ttFnt);
axis equal;
hold on;
r = logistic(sum(X(:, d), 2), ptTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(X(:, d1), X(:, d2), ptSiz, Rgb, 'filled');
set(gca, 'xtick', []);
set(gca, 'ytick', []);
box on;

subplot(4, 4, 5);
s = ksd(X, fu(X), k);
plot(1:nPart, log(s), 'color', spRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
xlabel('j', 'fontsize', lbFnt);
ylabel('log KSD', 'fontsize', lbFnt);
axis(axKsd);
box on;

subplot(4, 4, 9);
r = logistic(X(:, d1), tcTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(1:nPart, X(:, d1), tcSiz, Rgb, 'filled');
set(gca, 'fontsize', tkFnt);
xlabel('j', 'fontsize', lbFnt);
axis(axTrc);
box on;

subplot(4, 4, 13);
d2X = sum(diff(X, 1, 1) .^ 2, 2);
[fj2, j2] = ksdensity(d2X);
plot(j2, fj2, 'color', mcRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
xlabel('Jump^2', 'fontsize', lbFnt);
ylabel('Density', 'fontsize', lbFnt);
axis(axJmp);
box on;

% (2) LAST
X = xArr{2};
subplot(4, 4, 2);
contour(t1, t2, Z, ...
    'levelstep', lvStp, ...
    'linewidth', lnWdt, ...
    'linecolor', lnRgb);
title('LAST', 'fontsize', ttFnt);
axis equal;
hold on;
r = logistic(sum(X(:, d), 2), ptTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(X(:, d1), X(:, d2), ptSiz, Rgb, 'filled');
set(gca, 'xtick', []);
set(gca, 'ytick', []);
box on;

subplot(4, 4, 6);
s = ksd(X, fu(X), k);
plot(1:nPart, log(s), 'color', spRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
axis(axKsd);
box on;

subplot(4, 4, 10);
r = logistic(X(:, d1), tcTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(1:nPart, X(:, d1), tcSiz, Rgb, 'filled');
set(gca, 'fontsize', tkFnt);
axis(axTrc);
box on;

subplot(4, 4, 14);
load('fmin_grw_last_out.mat');
[d2X, d2Max, d2Sel, d2Lst] = jumpsize(X, Chain, obIdx);
[fj2, j2] = ksdensity(d2X);
plot(j2, fj2, 'color', spRgb, 'linewidth', lnWdt);
hold on;
[fj2, j2] = ksdensity(d2Max);
plot(j2, fj2, 'color', mcRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
axis(axJmp);
legend({'SP-MCMC', 'MCMC'}, 'location', 'best', 'fontsize', lgFnt);
box on;

% (3) RAND
X = xArr{3};
subplot(4, 4, 3);
contour(t1, t2, Z, ...
    'levelstep', lvStp, ...
    'linewidth', lnWdt, ...
    'linecolor', lnRgb);
title('RAND', 'fontsize', ttFnt);
axis equal;
hold on;
r = logistic(sum(X(:, d), 2), ptTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(X(:, d1), X(:, d2), ptSiz, Rgb, 'filled');
set(gca, 'xtick', []);
set(gca, 'ytick', []);
box on;

subplot(4, 4, 7);
s = ksd(X, fu(X), k);
plot(1:nPart, log(s), 'color', spRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
axis(axKsd);
box on;

subplot(4, 4, 11);
r = logistic(X(:, d1), tcTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(1:nPart, X(:, d1), tcSiz, Rgb, 'filled');
set(gca, 'fontsize', tkFnt);
axis(axTrc);
box on;

subplot(4, 4, 15);
load('fmin_grw_rand_out.mat');
[d2X, d2Max, d2Sel, d2Lst] = jumpsize(X, Chain, obIdx);
[fj2, j2] = ksdensity(d2X);
plot(j2, fj2, 'color', spRgb, 'linewidth', lnWdt);
hold on;
[fj2, j2] = ksdensity(d2Max);
plot(j2, fj2, 'color', mcRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
axis(axJmp);
box on;

% (4) INFL
X = xArr{4};
subplot(4, 4, 4);
contour(t1, t2, Z, ...
    'levelstep', lvStp, ...
    'linewidth', lnWdt, ...
    'linecolor', lnRgb);
title('INFL', 'fontsize', ttFnt);
axis equal;
hold on;
r = logistic(sum(X(:, d), 2), ptTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(X(:, d1), X(:, d2), ptSiz, Rgb, 'filled');
set(gca, 'xtick', []);
set(gca, 'ytick', []);
box on;

subplot(4, 4, 8);
s = ksd(X, fu(X), k);
plot(1:nPart, log(s), 'color', spRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
axis(axKsd);
box on;

subplot(4, 4, 12);
r = logistic(X(:, d1), tcTrn);
Rgb = [r, zeros(nPart, 1), 1 - r];
scatter(1:nPart, X(:, d1), tcSiz, Rgb, 'filled');
set(gca, 'fontsize', tkFnt);
axis(axTrc);
box on;

subplot(4, 4, 16);
load('fmin_grw_infl_out.mat');
[d2X, d2Max, d2Sel, d2Lst] = jumpsize(X, Chain, obIdx);
[fj2, j2] = ksdensity(d2X);
plot(j2, fj2, 'color', spRgb, 'linewidth', lnWdt);
hold on;
[fj2, j2] = ksdensity(d2Max);
plot(j2, fj2, 'color', mcRgb, 'linewidth', lnWdt);
set(gca, 'fontsize', tkFnt);
axis(axJmp);
box on;

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 25, 25]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 25, 25]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [25, 25]);

% Print to PDF
print('smry_gmm', '-dpdf');
