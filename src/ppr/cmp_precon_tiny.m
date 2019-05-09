%%
% File: cmp_precon_tiny.m
% Purpose:
% Compare point-sets generated with a preconditioned kernel to those
% generated without one. The target is a 'tiny' Gaussian distribution.
% This script creates Figure S2(b) in the supplement of SP-MCMC.
% Date: May 9, 2019
%%

% Add dependencies
addpath('../lib');

% Drop sites
nPts = 1000;
drop = false(nPts, 1);

% Target density and score functions
[Mu, C, w] = gmparam_trvl(0.01);
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Symbolic variables
nDim = 2;
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Run MCMC to obtain preconditioner
nSamp = 5000;
x0 = [0, 0];
% [Chain, ~, ~, acMala] = mala(fp, fu, x0, 0.15, eye(2), nSamp);
[Chain, acRw] = grwmetrop(fp, x0, 0.05 .* eye(nDim), nSamp);

% Preconditioned kernel
L = cov(Chain);
LInv = inv(L);
k1 = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% Identity kernel
L = eye(nDim);
LInv = inv(L);
k2 = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% Independent sample for computing energy distances
XInd = gaussmix_rnd(Mu, C, w, 20000);

% (1) INFL Pre.
nIter = repmat(5, nPts, 1);
x0 = [0, 0];
S0 = 0.05 .* eye(nDim);
lambda = 2.38 .^ 2 ./ 2;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl1] = stein_greedy_drop(nDim, fu, k1, fmin, drop);
X1 = X{end};
ed1 = mmd(X1, XInd);

% (2) INFL
nIter = repmat(5, nPts, 1);
x0 = [0, 0];
S0 = 0.05 .* eye(nDim);
lambda = 2.38 .^ 2 ./ 2;
alpha = ones(nPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ~, evl2] = stein_greedy_drop(nDim, fu, k2, fmin, drop);
X2 = X{end};
ed2 = mmd(X2, XInd);

% Energy distances vs number of target evaluations
lab = {'SP-MCMC Pre.', 'SP-MCMC'};
figure();
hold on;
for i = 1:2
    evl = eval(['evl', num2str(i)]);
    ed = eval(['ed', num2str(i)]);
    plot(log(cumsum(evl)), log(ed), 'linewidth', 1);
end
set(gca, 'fontsize', 16);
xlabel('log n_{eval}', 'fontsize', 16);
ylabel('log E_P', 'fontsize', 16);
axis([1.7, 9, -10, -2]);
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
print('ed_precon_tiny', '-dpdf');

% Save output
save('ed_precon_tiny.mat');
