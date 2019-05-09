function [h1, h2] = gaussmix_replay(X, SttPt, Mu, S, w, d, lb, ub, nStep, out)
% [h1, h2] = gaussmix_replay(X, SttPt, Mu, S, w, d, lb, ub, nStep, out)
% generates plots and video output for the greedy Stein algorithm with
% Markov transitions, where the target is a Gaussian mixture.
%
% Input:
% X     - nPart-by-nDim matrix of generated points.
% SttPt - nPart-by-nDim matrix of starting points.
% Mu    - nMix-by-nDim matrix of mean vectors.
% S     - nDim-by-nDim-by-nMix array of covariance matrices.
% w     - nMix-by-1 vector of weights.
% d     - 1-by-2 vector of the selected dimensions.
% lb    - 1-by-2 vector of rendering lower bounds.
% ub    - 1-by-2 vector of rendering upper bounds.
% nStep - grid resolution for plotting contours.
% out   - file name of the video output.
%
% Output:
% h1    - handle to the 'point set' plot.
% h2    - handle to the 'starting points' plot.
%
% Date: November 11, 2018

    % Number of points
    nPart = size(X, 1);

    % Keep only selected dimensions
    X = X(:, d);
    SttPt = SttPt(:, d);
    Mu = Mu(:, d);
    S = S(d, d, :);

    % Plot the contours and point set
    h1 = figure();
    t1 = linspace(lb(1), ub(1), nStep)';
    t2 = linspace(lb(2), ub(2), nStep)';
    T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
    p = fp_gaussmix(T, Mu, S, w);
    Z = reshape(p, nStep, nStep);
    contour(t1, t2, Z, ...
        'levelstep', 0.01, ...
        'linewidth', 1, ...
        'linecolor', [0.7, 0.7, 0.7]);
    xlabel(['x_', num2str(d(1))]);
    ylabel(['x_', num2str(d(2))]);
    title('Point Set');
    axis equal;
    hold on;
    plot(X(:, 1), X(:, 2), '.r', 'markersize', 12);

    % Plot starting points
    h2 = copyobj(h1, 0);
    plot(SttPt(:, 1), SttPt(:, 2), '-', 'color', [0.9, 0.5, 0.5]);
    title('Starting Points');
    axis equal;

    % Write to a video file
    h3 = figure();
    v = VideoWriter(out);
    v.FrameRate = 10;
    v.Quality = 50;
    open(v);
    for i = 1:nPart
        contour(t1, t2, Z, ...
            'levelstep', 0.01, ...
            'linewidth', 1, ...
            'linecolor', [0.7, 0.7, 0.7]);
        xlabel(['x_', num2str(d(1))]);
        ylabel(['x_', num2str(d(2))]);
        title('Point Sequence');
        axis equal;
        hold on;
        plot(X(1:(i - 1), 1), X(1:(i - 1), 2), '.r', 'markersize', 12);
        plot(SttPt(i, 1), SttPt(i, 2), 'ob', 'markersize', 10);
        writeVideo(v, getframe(h3));
        plot([SttPt(i, 1), X(i, 1)], [SttPt(i, 2), X(i, 2)], '-b');
        plot(X(i, 1), X(i, 2), '.r', 'markersize', 12);
        writeVideo(v, getframe(h3));
        hold off;
    end
    close(v);
end
