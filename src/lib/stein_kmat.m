function k0 = stein_kmat(X, D, k)
% k0 = stein_kmat(X, D, k) evaluates the Stein kernel matrix at X. The
% result is stored as a column vector of the upper-triangular elements
% of the Gram matrix.
%
% Input:
% X  - nObs-by-nDim matrix of nDim-dimensional points.
% D  - nIter-by-nDim matrix of scores at X.
% k  - symbolic expression of the kernel k(a,b), where a and b are 1-
%      by-nDim row vectors. It is important that the argument names are
%      literally "a" and "b".
%
% Output:
% k0 - column vector of the upper-triangular elements of the Stein kernel
%      matrix.
%
% Date: July 16, 2018

    % Dimensions
    [nObs, nDim] = size(X);

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

    % Evaluate the Stein kernel
    nElm = (nObs .^ 2 + nObs) ./ 2;
    [r, c] = i2rc(1:nElm);
    k0 = fk0(X(r, :), X(c, :), D(r, :), D(c, :));
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
