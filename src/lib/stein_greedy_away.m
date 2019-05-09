function [XArr, ksd, nEval, DArr, GArr] = stein_greedy_away( ...
    nDim, fscr, k, fmin, nPts, maxIter)
% [X, nEval, D, G] = stein_greedy_away(nDim, fscr, k, fmin, nPts, maxIter)
% generates a point sequence that approximates a target density by greedily
% minimising the KSD with 'away' steps.
%
% Input:
% nDim    - number of dimensions of the target density.
% fscr    - handle to the score function of the target density. The score
%           function must accept either an 1-by-nDim row vector or a n-by
%           -nDim matrix. It returns either an 1-by-nDim row vector or a
%           n-by-nDim matrix.
% k       - symbolic expression of the kernel k(a,b), where a and b are 1
%           -by-nDim row vectors. It is important that the argument names
%           are literally "a" and "b".
% fmin    - function handle to a nDim-dimensional minimiser.
% nPts    - maximum number of points to be generated.
% maxIter - maximum number of iterations.
%
% Output:
% XArr    - cell array of nPts-by-nDim matrices of generated points.
% ksd     - vector of KSD values over iterations.
% nEval   - number of score function evaluations at each iteration.
% DArr    - cell array of nPts-by-nDim matrices of scores.
% GArr    - cell array of nPts-by-nPts Stein kernel matrices.
%
% Date: January 7, 2019

    % Symbolic computations
    a = sym('a', [1, nDim], 'real');
    b = sym('b', [1, nDim], 'real');
    dka = sym(zeros(1, nDim));
    dkb = sym(zeros(1, nDim));
    d2k = sym(zeros(1, nDim));
    for i = 1:nDim
        dka(i) = gradient(k, a(i));
        dkb(i) = gradient(k, b(i));
        d2k(i) = gradient(dka(i), b(i));
    end

    % Generate MATLAB code
    matlabFunction(k, 'vars', {a, b}, 'file', 'fk.m');
    matlabFunction(dka, 'vars', {a, b}, 'file', 'fdka.m');
    matlabFunction(dkb, 'vars', {a, b}, 'file', 'fdkb.m');
    matlabFunction(d2k, 'vars', {a, b}, 'file', 'fd2k.m');

    % Generate x_1
    XArr = cell(maxIter, 1);
    DArr = cell(maxIter, 1);
    GArr = cell(maxIter, 1);
    ksd = zeros(maxIter, 1);
    nEval = zeros(maxIter, 1);
    X = zeros(nPts, nDim);
    D = zeros(nPts, nDim);
    L = zeros(nPts, nPts);
    f = @(XNew, DNew)fk0aa(XNew, DNew);
    [X(1, :), D(1, :), L(1, 1), nEval(1)] = fmin( ...
        f, double.empty(0, nDim), double.empty(0, nDim), fscr);
    ksd(1) = sqrt(L(1, 1));
    XArr{1} = X(1, :);
    DArr{1} = D(1, :);
    GArr{1} = L(1, 1);
    fprintf('n = 1\n');

    % Generate the rest
    n = 1;
    for i = 2:maxIter
        % Find least influential point
        G = mirror(L(1:n, 1:n));
        rowSum = sum(G, 2);
        colSum = sum(G, 1);
        infl = rowSum + colSum' - diag(G);
        [~, j] = max(infl);

        % New KSD after deleting j-th point
        nj = [1:(j - 1), (j + 1):n];
        ksdDel = sqrt(sum(sum(G(nj, nj)))) ./ (n - 1);

        % Find a new point to add
        f = @(XNew, DNew)fps(XNew, DNew, X, D, n + 1);
        [xAdd, dAdd, lAdd, nEval(i)] = fmin(f, X(1:n, :), G, fscr);

        % New KSD after adding a new point
        T = [[G; lAdd(1:n)], lAdd'];
        ksdAdd = sqrt(sum(sum(T))) ./ (n + 1);

        % Delete or add
        if (ksd(i - 1) - ksdDel) >= (ksd(i - 1) - ksdAdd)
            % Delete j-th point
            X(1:(n - 1), :) = X(nj, :);
            D(1:(n - 1), :) = D(nj, :);
            L(1:(n - 1), 1:(n - 1)) = L(nj, nj);
            ksd(i) = ksdDel;
            n = n - 1;
        else
            % Add a new point
            X(n + 1, :) = xAdd;
            D(n + 1, :) = dAdd;
            L(n + 1, 1:(n + 1)) = lAdd;
            ksd(i) = ksdAdd;
            n = n + 1;
        end

        % Save point-set state
        XArr{i} = X(1:n, :);
        DArr{i} = D(1:n, :);
        GArr{i} = mirror(L(1:n, 1:n));
        fprintf('n = %d\n', n);

        % Exit early if budget is reached
        if n >= nPts
            break
        end
    end

    % Trim output
    XArr = XArr(1:i);
    DArr = DArr(1:i);
    GArr = GArr(1:i);
    ksd = ksd(1:i);
    nEval = nEval(1:i);
end

function [ps, K0] = fps(XNew, DNew, X, D, n)
% Input:
% XNew - nNew-by-nDim matrix of new points.
% DNew - nNew-by-nDim score matrix of the target at XNew.
% X    - nObs-by-nDim matrix of current sample.
% D    - nObs-by-nDim matrix of scores at X.
% n    - observation index of XNew.
%
% Output:
% ps   - nNew-by-1 column vector of partial sums for XNew.
% K0   - nNew-by-n matrix of Stein kernel values.

    nNew = size(XNew, 1);
    A = repmat(XNew, n - 1, 1);
    B = repelem(X(1:(n - 1), :), nNew, 1);
    Da = repmat(DNew, n - 1, 1);
    Db = repelem(D(1:(n - 1), :), nNew, 1);
    K0ab = reshape(fk0(A, B, Da, Db), nNew, []);
    k0aa = fk0(XNew, XNew, DNew, DNew);
    ps = sum(K0ab, 2) .* 2 + k0aa;
    K0 = [K0ab, k0aa];
end

function [k0aa, k0] = fk0aa(XNew, DNew)
% Input:
% XNew - nNew-by-nDim matrix of new points.
% DNew - nNew-by-nDim score matrix of the target at XNew.
%
% Output:
% k0aa - nNew-by-1 column vector of Stein kernel values.
% K0   - nNew-by-1 column vector identical to k0aa.

    k0aa = fk0(XNew, XNew, DNew, DNew);
    k0 = k0aa;
end

function k0 = fk0(A, B, Da, Db)
% Input:
% A  - m-by-nDim matrix of the first arguments.
% B  - m-by-nDim matrix of the second arguments.
% Da - m-by-nDim matrix of scores at A.
% Db - m-by-nDim matrix of scores at B.
%
% Output:
% k0 - m-by-1 column vector of Stein kernel values.

    nDim = size(A, 2);
    K0i = ...
        fd2k(A, B) + ...
        Da .* fdkb(A, B) + ...
        Db .* fdka(A, B) + ...
        Da .* Db .* repmat(fk(A, B), 1, nDim);
    k0 = sum(K0i, 2);
end
