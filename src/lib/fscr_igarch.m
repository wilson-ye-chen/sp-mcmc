function D = fscr_igarch(Tht, y, h1)
% D = fscr_igarch(Tht, y, h1) evaluates the score function of the posterior
% of the zero-mean Gaussian-IGARCH(1,1) model.
%
% Input:
% Tht - n-by-2 matrix of n sets of IGARCH parameters.
% y   - vector of observations.
% h1  - scalar of initial variance.
%
% Output:
% D   - n-by-2 matrix of score function values.
%
% Date: October 31, 2017

    n = size(Tht, 1);
    nObs = numel(y);
    tht1 = Tht(:, 1);
    tht2 = Tht(:, 2);

    dh_tht1 = zeros(n, 1);
    dh_tht2 = zeros(n, 1);
    h = ones(n, 1) .* h1;
    dp_tht1 = zeros(n, 1);
    dp_tht2 = zeros(n, 1);
    ySq = y .^ 2;
    b = 1 - tht2;
    for i = 2:nObs
        dh_tht1 = 1 + b .* dh_tht1;
        dh_tht2 = ySq(i - 1) - h + b .* dh_tht2;
        h = tht1 + tht2 .* ySq(i - 1) + b .* h;
        dp_h = -1 ./ (2 .* h) + ySq(i) ./ (2 .* h .^ 2);
        dp_tht1 = dp_tht1 + dp_h .* dh_tht1;
        dp_tht2 = dp_tht2 + dp_h .* dh_tht2;
    end
    D = [dp_tht1, dp_tht2];
end
