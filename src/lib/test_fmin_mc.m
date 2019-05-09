%%
% File: test_fmin_mc.m
% Purpose:
% Runs the greedy Stein algorithm with Markov transitions.
% Date: January 18, 2019
%%

% Test case
testCase = 1;
if testCase == 1
    [Mu, C, w] = gmparam_2c(2, 1.5);
elseif testCase == 2
    rng(8);
    [Mu, C, w] = gmparam_rnd();
    rng('shuffle');
end

% Number of points
nPts = 1000;

% Target distribution
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Target dimension
nDim = size(Mu, 2);

% Optimiser
nMc = 20;
mu0 = ones(1, nDim);
mu0(1:2:end) = -1;
S0 = 5 .* eye(nDim);
lambda = 0.5;
alpha = 1 - logistic(linspace(-1, 1, nPts), 8);
fmin = @(f, X, G, fscr)fmin_mc( ...
    f, X, G, fscr, nMc, mu0, S0, lambda, alpha);

% Kernels
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');
ll = median(pdst(gaussmix_rnd(Mu, C, w, nPts))) .^ 2 ./ log(nPts);
k = (1 + (a - b) * (a - b)' ./ ll) .^ (-0.5);

% SP 'Standard'
drop = false(nPts, 1);
[X1, ks1, nEval1] = stein_greedy_drop(nDim, fu, k, fmin, drop);
n1 = cellfun(@(A)size(A, 1), X1);

% SP 'Drop'
drop = false(nPts + 300, 1);
drop(round(linspace(10, nPts + 300, 300))) = true;
[X2, ks2, nEval2] = stein_greedy_drop(nDim, fu, k, fmin, drop);
n2 = cellfun(@(A)size(A, 1), X2);

% SP 'Refine'
drop = false(nPts + 300, 1);
drop((nPts + 1):(nPts + 300)) = true;
[X3, ks3, nEval3] = stein_greedy_drop(nDim, fu, k, fmin, drop);
n3 = cellfun(@(A)size(A, 1), X3);

% SP 'Away'
[X4, ks4, nEval4] = stein_greedy_away(nDim, fu, k, fmin, nPts, nPts + 300);
n4 = cellfun(@(A)size(A, 1), X4);

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
subplot(2, 2, 1);
contour(t1, t2, Z, ...
    'levelstep', 0.01, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('x_1');
ylabel('x_2');
title('Standard');
axis equal;
hold on;
X = X1{end};
plot(X(:, 1), X(:, 2), '.r', 'markersize', 8);

% Contour 'Drop'
subplot(2, 2, 2);
contour(t1, t2, Z, ...
    'levelstep', 0.01, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('x_1');
ylabel('x_2');
title('Drop');
axis equal;
hold on;
X = X2{end};
plot(X(:, 1), X(:, 2), '.r', 'markersize', 8);

% Contour 'Refine'
subplot(2, 2, 3);
contour(t1, t2, Z, ...
    'levelstep', 0.01, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('x_1');
ylabel('x_2');
title('Refine');
axis equal;
hold on;
X = X3{end};
plot(X(:, 1), X(:, 2), '.r', 'markersize', 8);

% Contour 'Away'
subplot(2, 2, 4);
contour(t1, t2, Z, ...
    'levelstep', 0.01, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('x_1');
ylabel('x_2');
title('Away');
axis equal;
hold on;
X = X4{end};
plot(X(:, 1), X(:, 2), '.r', 'markersize', 8);

% KSD vs number of target evaluations
figure();
hold on;
plot(log(cumsum(nEval1)), log(ks1), 'linewidth', 1);
plot(log(cumsum(nEval2)), log(ks2), 'linewidth', 1);
plot(log(cumsum(nEval3)), log(ks3), 'linewidth', 1);
plot(log(cumsum(nEval4)), log(ks4), 'linewidth', 1);
xlabel('log n_{eval}');
ylabel('log KSD');
legend({'Standard', 'Drop', 'Refine', 'Away'}, ...
    'fontsize', 13, ...
    'location', 'best');

% KSD vs number of iterations
figure();
hold on;
plot(log(ks1), 'linewidth', 1);
plot(log(ks2), 'linewidth', 1);
plot(log(ks3), 'linewidth', 1);
plot(log(ks4), 'linewidth', 1);
xlabel('Iteration');
ylabel('log KSD');
legend({'Standard', 'Drop', 'Refine', 'Away'}, ...
    'fontsize', 13, ...
    'location', 'best');

% Growth 'Standard'
figure();
subplot(2, 2, 1);
plot(n1, 'linewidth', 1);
axis([0, 1300, 0, 1100]);
xlabel('Iteration');
ylabel('n');
title('Standard');

% Growth 'Drop'
subplot(2, 2, 2);
plot(n2, 'linewidth', 1);
axis([0, 1300, 0, 1100]);
xlabel('Iteration');
ylabel('n');
title('Drop');

% Growth 'Refine'
subplot(2, 2, 3);
plot(n3, 'linewidth', 1);
axis([0, 1300, 0, 1100]);
xlabel('Iteration');
ylabel('n');
title('Refine');

% Growth 'Away'
subplot(2, 2, 4);
plot(n4, 'linewidth', 1);
axis([0, 1300, 0, 100]);
xlabel('Iteration');
ylabel('n');
title('Away');
