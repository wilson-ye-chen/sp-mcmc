function [J, iSub] = contour_aprx(X, lb, ub, nStep, lvstp, lnwth)
% [J, iSub] = contour_aprx(X, lb, ub, nStep, lvstp, lnwth) creates contour
% plots of the pair-wise marginal densities based on the point-set matrix X.
%
% Date: 25 June, 2018

    % Upper-triangular index set
    nDim = size(X, 2);
    [J1, J2] = meshgrid(1:nDim, 1:nDim);
    J = [J1(:), J2(:)];
    J = J(J(:, 1) < J(:, 2), :);

    % Iterate over the upper-triangular pairs
    nElm = (nDim .^ 2 - nDim) ./ 2;
    iSub = zeros(nElm, 1);
    for i = 1:nElm
        % Create a grid
        st1 = linspace(lb(J(i, 1)), ub(J(i, 1)), nStep);
        st2 = linspace(lb(J(i, 2)), ub(J(i, 2)), nStep);
        [T1, T2] = meshgrid(st1, st2);
        T = [T1(:), T2(:)];

        % Approximate the pair-wise marginal density
        p = ksdensity([X(:, J(i, 1)), X(:, J(i, 2))], T);
        Z = reshape(p, nStep, nStep);

        % Plot the contours
        iSub(i) = (J(i, 1) - 1) .* nDim + J(i, 2) - J(i, 1);
        subplot(nDim - 1, nDim - 1, iSub(i));
        contour(T1, T2, Z, 'levelstep', lvstp, 'linewidth', lnwth);
        set(gca, 'visible', 'off');

        % Print progress
        fprintf('[%.0f%%]', i ./ nElm .* 100);
    end
    fprintf('\n');
end
