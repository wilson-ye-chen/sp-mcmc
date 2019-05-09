%%
% File: gensmry_drop.m
% Purpose:
% Generates summary plots for INFL with various drop rates.
% The target is a two-dimensional standard Gaussian mixture model.
% Date: January 8, 2019
%%

% Add dependencies
addpath('../lib');

% Number of particles
nPart = 1000;

% Target density and score functions
[Mu, C, w] = gmparam_2c(2, 1.5);
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Symbolic variables
nDim = size(Mu, 2);
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');

% Median trick based on independent sample
ll = median(pdst(gaussmix_rnd(Mu, C, w, nPart))) .^ 2 ./ log(nPart);
k = (1 + (a - b) * (a - b)' ./ ll) .^ (-0.5);

% MCMC optimiser setup
nIter = 5;
x0 = zeros(1, nDim);
S0 = diag(0.1 .* ones(1, nDim));
lambda = 0.1;
alpha = ones(nPart, 1);

% INFL
fmin = @(f, X, G, fscr)fmin_grw_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, true);

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
axStp = [-0.2, 6.5, 0, 1.5];
lvStp = 0.01;
lnWdt = 1;
lnClr = [0.7, 0.7, 0.7];
mkSiz = 5;

% Run INFL with various drop rates
dr = [0, 0.1, 0.3];
mk = {'o', '^', 's'};
for i = 1:numel(dr)
    % Delete existing output file
    fminOut = 'fmin_grw_infl_out.mat';
    delete(fminOut);
    % Generate points
    nGen = round(nPart ./ (1 - dr(i)));
    drop = false(nGen, 1);
    drop(round(linspace(10, nGen, nGen - nPart))) = true;
    [X, nEval, D] = stein_greedy_drop(nDim, fu, k, fmin, drop);
    % Generate plot
    subplot(2, 2, i);
    contour(t1, t2, Z, ...
        'levelstep', lvStp, ...
        'linewidth', lnWdt, ...
        'linecolor', lnClr);
    xlabel(['x_', num2str(d(1))]);
    ylabel(['x_', num2str(d(2))]);
    title(['Drop rate = ', num2str(dr(i))]);
    axis equal;
    hold on;
    load(fminOut);
    plot(MinPt(:, d(1)), MinPt(:, d(2)), '.r', 'markersize', mkSiz);
    DelPt = setdiff(MinPt, X, 'stable', 'rows');
    plot(DelPt(:, d(1)), DelPt(:, d(2)), 'bx', 'markersize', mkSiz);
    % Plot KSD
    subplot(2, 2, 4);
    hold on;
    plot(1:size(X, 1), log(ksd(X, D, k)), mk{i});
end

% Add labels for the KSD plot
xlabel('n');
ylabel('log KSD');
legend('0', '0.1', '0.3', 'location', 'best');
