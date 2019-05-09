function A = vechinv(v)
% A = vechinv(v) constructs an upper-triangular matrix by sequentially
% placing each element of a vector column-wise.
%
% Input:
% v - vector of length (n^2+n)/2.
%
% Output:
% A - n-by-n matrix whose upper-triangular elements are elements of v.
%
% Date: August 1, 2018

    [r, c] = i2rc(1:numel(v));
    n = r(end);
    A = zeros(n, n);
    A(sub2ind([n, n], r, c)) = v;
end
