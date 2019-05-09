function p = fp_igarch(Tht, y, h1)
% p = fp_igarch(Tht, y, h1) evaluates the un-normalised log-posterior
% density of the posterior of the zero-mean Gaussian-IGARCH(1,1) model.
%
% Input:
% Tht - n-by-2 matrix of n sets of IGARCH parameters.
% y   - vector of observations.
% h1  - scalar of initial variance.
%
% Output:
% D   - n-by-1 vector of log-posterior values.
%
% Date: October 31, 2017

    n = size(Tht, 1);
    nObs = numel(y);
    tht1 = Tht(:, 1);
    tht2 = Tht(:, 2);

    h = ones(n, 1) .* h1;
    p = zeros(n, 1);
    for i = 2:nObs
        h = tht1 + tht2 .* y(i - 1) .^ 2 + (1 - tht2) .* h;
        p = p + log(normpdf(y(i), 0, sqrt(h)));
    end
end
