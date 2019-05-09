function v = vech(A)
% v = vech(A) constructs a column vector by stacking column-wise
% the upper-triangular elements of a square matrix.
%
% Input:
% A - n-by-n square matrix.
%
% Output:
% v - column vector of length (n^2+n)/2 containing the upper-
%     triangular elements of A.
%
% Date: August 1, 2018

    n = size(A, 1);
    [r, c] = i2rc([1:((n .^ 2 + n) ./ 2)]');
    v = A(sub2ind([n, n], r, c));
end
