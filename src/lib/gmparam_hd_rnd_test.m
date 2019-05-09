%%
% File: gmparam_hd_rnd_test.m
% Purpose:
% Creates a grid of contour plots of the pair-wise marginal densities of
% a randomly generated higher-dimensional Gaussian mixture distribution.
% Date: June 25, 2018
%%

% Generate random parameter values
[Mu, S, w] = gmparam_hd_rnd();

% Sample from this distribution
nObs = 200000;
X = gaussmix_rnd(Mu, S, w, nObs);

% Contour plots of pair-wise marginals
nDim = size(X, 2);
lb = -8 .* ones(1, nDim);
ub = 8 .* ones(1, nDim);
contour_aprx(X, lb, ub, 100, 0.001, 0.5);

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 40, 40]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 40, 40]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [40, 40]);

% Print to PDF
print('gmparam_hd_rnd_test_out', '-dpdf');
