function S = mirror(L)
% S = mirror(L) creates a symmetric matrix by mirroring
% a lower-triangular along its diagonal.
%
% Date: June 27, 2018

    if ~istril(L)
        error('Argument is not lower-triangular.');
        return
    end

    n = size(L, 1);
    S = L + L';
    S(1:(n + 1):end) = diag(L);
end
