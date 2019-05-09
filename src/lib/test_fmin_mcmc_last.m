%%
% File: test_fmin_mcmc_last.m
% Purpose:
% Runs the greedy Stein algorithm with Markov transitions.
% Date: December 12, 2018
%%

% Test case
testCase = 1;
if testCase == 1
    [Mu, C, w] = gmparam_2c();
elseif testCase == 2
    rng(8);
    [Mu, C, w] = gmparam_rnd();
    rng('shuffle');
end

% Target distribution
fp = @(X)log(fp_gaussmix(X, Mu, C, w));
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Budget
nPart = 500;

% Optimiser
nIter = repmat(5, nPart, 1);
x0 = [0, 0];
S0 = diag([0.1, 0.1]);
lambda = 0.1;
alpha = ones(nPart, 1);
lb = [-10, -10];
ub = [10, 10];
fpcon = @(x)fp(x) + logind(x, lb, ub, [], []);
fmin = @(f, X, G, fscr)fmin_grw_last( ...
    f, X, G, fscr, fpcon, nIter, x0, S0, lambda, alpha, true);

% Kernels
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');
ll = 1 .^ 2;
k = (1 + (a - b) * (a - b)' ./ ll) .^ (-0.5);

% Run Stein greedy
name = 'fmin_grw_last_out';
delete([name, '.mat']);
tic;
[X, nEval, D, G] = stein_greedy(2, fu, k, fmin, nPart);
toc;
load([name, '.mat']);

% Plot the contours and point set
fig1 = figure();
nStep = 100;
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = exp(fp(T));
Z = reshape(p, nStep, nStep);
contour(t1, t2, Z, ...
    'levelstep', 0.001, ...
    'linewidth', 1, ...
    'linecolor', [0.7, 0.7, 0.7]);
xlabel('x_1');
ylabel('x_2');
title('LAST - Point Set');
hold on;
plot(X(:, 1), X(:, 2), '.r', 'markersize', 12);

% Plot starting points
fig2 = copyobj(fig1, 0);
plot(SttPt(:, 1), SttPt(:, 2), '-', 'color', [0.9, 0.5, 0.5]);
txt = cellstr(num2str([1:nPart]'));
text(SttPt(:, 1), SttPt(:, 2), txt);
title('LAST - Starting Points');

% Write to a video file
fig3 = figure();
v = VideoWriter([name, '.avi']);
v.FrameRate = 10;
v.Quality = 50;
open(v);
for i = 1:nPart
    contour(t1, t2, Z, ...
        'levelstep', 0.001, ...
        'linewidth', 1, ...
        'linecolor', [0.7, 0.7, 0.7]);
    xlabel('x_1');
    ylabel('x_2');
    title('LAST - Point Sequence');
    hold on;
    plot(X(1:(i - 1), 1), X(1:(i - 1), 2), '.r', 'markersize', 12);
    plot(SttPt(i, 1), SttPt(i, 2), 'ob', 'markersize', 10);
    plot(SttPt(i, 1), SttPt(i, 2), 'ob', 'markersize', 15);
    writeVideo(v, getframe(fig3));
    plot([SttPt(i, 1), X(i, 1)], [SttPt(i, 2), X(i, 2)], '-b');
    plot(X(i, 1), X(i, 2), '.r', 'markersize', 12);
    writeVideo(v, getframe(fig3));
    hold off;
end
close(v);
