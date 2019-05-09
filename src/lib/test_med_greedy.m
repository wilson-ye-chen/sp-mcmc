%%
% File: test_med_greedy.m
% Purpose:
% Runs the greedy MED algorithm with Nelder-Mead optimiser.
% Date: January 18, 2019
%%

% Test case
testCase = 1;
if testCase == 1
    [Mu, C, w] = gmparam_2c(2, 0.5);
elseif testCase == 2
    [Mu, C, w] = gmparam_rnd();
end

% Number of points
nPts = 1000;

% Target distribution
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Target dimension
nDim = size(Mu, 2);

% Optimiser parameters
nInit = 3;
mu0 = ones(1, nDim);
mu0(1:2:end) = -1;
S0 = 5 .* eye(nDim);
lambda = 0.5;
alpha = 1 - logistic(linspace(-1, 1, nPts), 8);

% Kernels
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');
ll = median(pdst(gaussmix_rnd(Mu, C, w, nPts))) .^ 2 ./ log(nPts);
k = (1 + (a - b) * (a - b)' ./ ll) .^ (-0.5);

% MED greedy
fmin = @(f, X)fmin_nm(f, X, nInit, mu0, S0, lambda, alpha);
[X, negOb, nEval] = med_greedy(nDim, fp, fmin, nPts);

% Evaluate target density over a grid
lb = [-3.5, -3.5];
ub = [3.5, 3.5];
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = fp_gaussmix(T, Mu, C, w);
Z = reshape(p, nStep, nStep);

% Contour 'Standard'
figure();
contour(t1, t2, Z, ...
    'levelstep', 0.01, ...
    'linewidth', 1, ...
    'linecolor', [0.6, 0.6, 0.6]);
title('MED');
axis equal;
hold on;
plot(X(:, 1), X(:, 2), '.r', 'markersize', 8);

% KSD vs number of target evaluations
figure();
ks = ksd(X, fu(X), k);
plot(log(cumsum(nEval)), log(ks), 'linewidth', 1);
xlabel('log n_{eval}');
ylabel('log KSD');

% Negative MED objective function values
figure();
plot(negOb, 'linewidth', 1);
xlabel('Iteration');
ylabel('Negative MED objective');
