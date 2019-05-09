function [X, nEval] = mysvgd(fscr, k, X0, fstp, nIter)
% [X, nEval] = mysvgd(fscr, k, X0, fstp, nIter) iteratively moves a set
% of initial particles to match the target distribution using the Stein
% variational gradient descent (SVGD) algorithm. This main feature of
% this version is the ability to have a symbolically defined user-supplied
% kernel. As a consequence, adaptive bandwidth of the kernel is not
% supported. During runtime, this function generates 'fk.m' and 'fdka.m'
% code files in the directory from which it is called.
%
% Input:
% fscr  - handle to the score function of the target density. The score
%         function must accept either an 1-by-nDim row vector or a n-by
%         -nDim matrix. It returns either an 1-by-nDim row vector or a
%         n-by-nDim matrix.
% k     - symbolic expression of the kernel k(a,b), where a and b are 1
%         -by-nDim row vectors. It is important that the argument names
%         are literally "a" and "b".
% X0    - nPar-by-nDim matrix of the nDim-dimensional initial particles,
%         where nPar is the number of particles.
% fstp  - handle to the step-size function. The step-size function must
%         accept an iteration index as its first argument, and a nPar-by
%         -nDim matrix characterising the perturbation direction as its
%         second argument. It must return either a scalar or a nPar-by-
%         nDim matrix.
% nIter - number of iterations.
%
% Output:
% X     - nPar-by-nDim-by-nIter array of particles.
% nEval - nIter-by-1 vector of number of score function evaluations.
%
% Date: October 12, 2017

    % Symbolic computations
    [nPar, nDim] = size(X0);
    a = sym('a', [1, nDim], 'real');
    b = sym('b', [1, nDim], 'real');
    dka = gradient(k, a)';

    % Generate MATLAB code
    matlabFunction(k, 'vars', {a, b}, 'file', 'fk.m');
    matlabFunction(dka, 'vars', {a, b}, 'file', 'fdka.m');

    % Move particles
    X = zeros(nPar, nDim, nIter);
    nEval = zeros(nIter, 1);
    X(:, :, 1) = X0;
    for i = 1:(nIter - 1)
        D = fscr(X(:, :, i));
        Phi = fphi(X(:, :, i), D, nPar, nDim);
        X(:, :, i + 1) = X(:, :, i) + fstp(i, Phi) .* Phi;
        nEval(i + 1) = nPar;
    end
end

function Phi = fphi(X, D, nPar, nDim)
% Input:
% X    - nPar-by-nDim matrix of particles.
% nPar - number of particles.
% nDim - number of dimensions of the target density.
% fscr - handle to the score function of the target density.
%
% Output:
% Phi  - nPar-by-nDim matrix of perturbation directions.

    A = repmat(X, nPar, 1);
    B = repelem(X, nPar, 1);
    D = repmat(D, nPar, 1);
    Sk = repmat(fk(A, B), 1, nDim) .* D + fdka(A, B);
    Phi = splitapply(@mean, Sk, repelem([1:nPar]', nPar));
end
