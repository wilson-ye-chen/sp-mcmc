function fdraw_2d(f, xNew, l, u, n)
% fdraw_2d(f, xNew, l, u, n) plots the contours of a bivariate
% function f, and marks the point xNew.
%
% Input:
% f    - function handle to the objective.
% xNew - 2-vector of marker coordinates.
% l    - 2-vector of lower bounds of the search grid.
% u    - 2-vector of upper bounds of the search grid.
% n    - 2-vector of grid resolutions.
%
% Date: October 29, 2017

    st1 = linspace(l(1), u(1), n(1));
    st2 = linspace(l(2), u(2), n(2));
    [T1, T2] = meshgrid(st1, st2);
    T = [T1(:), T2(:)];
    val = f(T);
    Z = reshape(val, n(1), n(2));
    surf(T1, T2, Z, 'edgecolor', 'none');
    view(2);
    axis equal;
    hold on;
    plot3(xNew(1), xNew(2), max(val(val < inf)), ...
        'rx', 'markersize', 8, 'linewidth', 2);
    hold off;
    input('Press <Enter> to continue...');
end
