function [R, C] = i2rc(I)
% [R, C] = i2rc(I) computes the row and column indices of a
% matrix given the column-major linear index of its upper-
% triangular elements. The linear index 'I' can be either a
% scalar, vector, or matrix.
%
% Date: July 13, 2018

    C = ceil(sqrt(0.25 + 2 .* I) - 0.5);
    N = (C .^ 2 + C) ./ 2;
    R = C - N + I;
end
