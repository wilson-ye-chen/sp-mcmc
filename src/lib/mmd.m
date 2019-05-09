function e = mmd(X, Y)
% e = mmd(X, Y) returns a cumulative sequence of n energy distances.
%
% Input:
% X - n-by-nDim matrix of nDim-dimensional points.
% Y - m-by-nDim matrix of nDim-dimensional points.
%
% Output:
% e - n-by-1 vector of energy distances.
%
% Date: January 23, 2019

    nMax = size(X, 1);
    m = size(Y, 1);
    yy = 2 .* sum(pdist(Y)) ./ (m .^ 2);
    e = zeros(nMax, 1);
    yxSum = 0;
    xxSum = 0;
    for n = 1:nMax
        d = l2(Y, repmat(X(n, :), m, 1));
        yxSum = yxSum + sum(d);
        d = l2(X(1:(n - 1), :), repmat(X(n, :), n - 1, 1));
        xxSum = xxSum + 2 .* sum(d);
        yx = 2 .* yxSum ./ m ./ n;
        xx = xxSum ./ (n .^ 2);
        e(n) = yx - yy - xx;
    end
end

function d = l2(A, B)
    d = sqrt(sum((A - B) .^ 2, 2));
end
