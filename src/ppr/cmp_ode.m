%%
% File: cmp_ode.m
% Purpose:
% Compare point-sets from various discrete approximation algorithms.
% The target is the posterior of a Goodwin oscillator. This script
% creates Figures 4 and 5 in SP-MCMC.
% Date: May 9, 2019
%%

% Add dependencies
addpath('../lib');

% Test cases
testCase = 2;
if testCase == 1
    g = 2;
    numPts = 1000;
    lenScl = 0.3;
    chnLen = 10;
    prpScl = 2.38 .^ 2 ./ 4;
    malStp = 1.2;
    svgItr = 200;
    svgStp = 5e-3;
    numSmp = 10;
    iniVar = 0.05;
    mixVar = 0.001;
    preCon = 'precon_ode_4d.mat';
    axsLim = [2, 12.3, -1.2, 5];
elseif testCase == 2
    g = 8;
    numPts = 1000;
    lenScl = 0.15;
    chnLen = 20;
    prpScl = 2.38 .^ 2 ./ 10;
    malStp = 0.07;
    svgItr = 200;
    svgStp = 5e-3;
    numSmp = 20;
    iniVar = 0.05;
    mixVar = 0.001;
    preCon = 'precon_ode_10d.mat';
    axsLim = [2.9, 13, 0.5, 8.1];
end

% Drop sites
drop = false(numPts, 1);

% Target density and score functions
fp = @(X)goodwin(X, g);
fu = @(X)out2(@()goodwin(X, g));

% Target dimension
nDim = g + 2;

% Run adaptive Metropolis to re-generate preconditioner
% h0 = 2.38 ./ sqrt(nDim);
% epoch = [repmat(500, 20, 1); repmat(1000, 10, 1); 5000; 10000; 20000];
% alpha = 1 - logistic(linspace(-0.1, 1.9, numel(epoch)), 20);
% [hRw, ~, XRw, acRw] = grwmetrop_adapt(fp, x0, h0, eye(nDim), alpha, epoch);
% V = cov(XRw{end});
% save(preCon, 'V');

% Load preconditioner 'V'
load(preCon);

% Symbolic variables
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Kernel
L = lenScl .* eye(nDim);
LInv = inv(L);
k = (1 + (a - b) * LInv * (a - b)') .^ (-0.5);

% (1) MALA
nThin = chnLen;
nIter = numPts .* nThin;
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
[X, D, ~, acMala] = mala(fp, fu, x0, malStp, V, nIter);
X1 = X(1:nThin:end, :);
D1 = D(1:nThin:end, :);
ks1 = ksd(X1, D1, k);
evl1 = repmat(2 .* nThin, numPts, 1);
disp('MALA Done.');

% (2) GRW
nThin = chnLen;
nIter = numPts .* nThin;
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
[X, acRw] = grwmetrop(fp, x0, prpScl .* V, nIter);
X2 = X(1:nThin:end, :);
ks2 = ksd(X2, fu(X2), k);
evl2 = repmat(nThin, numPts, 1);
disp('RWM Done.');

% (3) SVGD
C = repmat(log([1, 3, 2, ones(1, g - 2), 0.5]), numPts, 1);
X0 = unifrnd(C - 0.5, C + 0.5);
fstp = @(i, Phi)fstp_adagrad(i, Phi, svgStp, 0.9);
[X, evl3] = mysvgd(fu, k, X0, fstp, svgItr);
X3 = X(:, :, end);
ks3 = zeros(svgItr, 1);
for i = 1:svgItr
    ks = ksd(X(:, :, i), fu(X(:, :, i)), k);
    ks3(i) = ks(end);
end

% (4) MED
mu0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
S0 = iniVar .* eye(nDim);
alpha = 1 - logistic(linspace(-0.35, 1.65, numPts), 30);
alpha(1) = 1;
fmin = @(f, X)fmin_mc_med(f, X, numSmp, mu0, S0, mixVar, alpha);
[X4, ~, evl4] = med_greedy(nDim, fp, fmin, numPts);
ks4 = ksd(X4, fu(X4), k);

% (5) SP
mu0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
S0 = iniVar .* eye(nDim);
alpha = 1 - logistic(linspace(-0.35, 1.65, numPts), 30);
alpha(1) = 1;
fmin = @(f, X, G, fscr)fmin_mc( ...
    f, X, G, fscr, numSmp, x0, S0, mixVar, alpha);
[X, ks5, evl5] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X5 = X{end};

% (6) SP-MALA (LAST)
nIter = repmat(chnLen, numPts, 1);
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
alpha = ones(numPts, 1);
fmin = @(f, X, G, fscr)fmin_mala_last( ...
    f, X, G, fscr, fp, nIter, x0, V, malStp, alpha, false);
[X, ks6, evl6] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X6 = X{end};

% (7) SP-MALA (INFL)
nIter = repmat(chnLen, numPts, 1);
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
alpha = ones(numPts, 1);
fmin = @(f, X, G, fscr)fmin_mala_infl( ...
    f, X, G, fscr, fp, nIter, x0, V, malStp, alpha, false);
[X, ks7, evl7] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X7 = X{end};

% (8) SP-GRW (LAST)
nIter = repmat(chnLen, numPts, 1);
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
S0 = prpScl .* V;
lambda = 2.38 .^ 2 ./ nDim;
alpha = ones(numPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ks8, evl8] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X8 = X{end};

% (9) SP-GRW (INFL)
nIter = repmat(chnLen, numPts, 1);
x0 = log([1, 3, 2, ones(1, g - 2), 0.5]);
S0 = prpScl .* V;
lambda = 2.38 .^ 2 ./ nDim;
alpha = ones(numPts, 1);
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, false);
[X, ks9, evl9] = stein_greedy_drop(nDim, fu, k, fmin, drop);
X9 = X{end};

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
axis(axsLim);
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

% Print to PDF and save output
if testCase == 1
    name = 'ksd_ode_4d';
elseif testCase == 2
    name = 'ksd_ode_10d';
end
print(name, '-dpdf');
save([name, '.mat']);
