%%
% File: test_fmin_mcmc_infl.m
% Purpose:
% Runs the greedy Stein algorithm with Markov transitions.
% Date: February 1, 2019
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

% Target distribution
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Target dimension
nDim = size(Mu, 2);

% Optimiser
nIter = repmat(5, 1000, 1);
x0 = ones(1, nDim);
x0(1:2:end) = -1;
S0 = 1.1 .* eye(nDim);
h = 1;
alpha = ones(1000, 1);
fmin = @(f, X, G, fscr)fmin_mala_infl( ...
    f, X, G, fscr, fp, nIter, x0, S0, h, alpha, false);

% Kernels
a = sym('a', [1, nDim], 'real');
b = sym('b', [1, nDim], 'real');
ll = median(pdst(gaussmix_rnd(Mu, C, w, 1000))) .^ 2 ./ log(1000);
k = (1 + (a - b) * (a - b)' ./ ll) .^ (-0.5);

% SP 'Standard'
drop = false(1000, 1);
[X1, ks1, nEval1, D1, G1] = stein_greedy_drop(nDim, fu, k, fmin, drop);
n1 = cellfun(@(A)size(A, 1), X1);

% SP 'Drop'
drop = false(1300, 1);
drop(round(linspace(10, 1300, 300))) = true;
[X2, ks2, nEval2, D2, G2] = stein_greedy_drop(nDim, fu, k, fmin, drop);
n2 = cellfun(@(A)size(A, 1), X2);

% SP 'Refine'
drop = false(1300, 1);
drop(1001:1300) = true;
[X3, ks3, nEval3, D3, G3] = stein_greedy_drop(nDim, fu, k, fmin, drop);
n3 = cellfun(@(A)size(A, 1), X3);

% SP 'Away'
[X4, ks4, nEval4, D4, G4] = stein_greedy_away(nDim, fu, k, fmin, 1000, 1300);
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
