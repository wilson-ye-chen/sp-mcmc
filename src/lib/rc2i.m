function I = rc2i(R, C)
% I = rc2i(R, C) computes the column-major linear index of
% an upper-triangular matrix. The row and column indices R
% and C can be either scalars, vectors, or matrices of the
% same size.
%
% Date: July 13, 2018

    N = (C .^ 2 + C) ./ 2;
    I = N - C + R;
end
