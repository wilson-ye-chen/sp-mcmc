function [p, D] = goodwin(Q, g)
% [p, D] = goodwin(Q, g) is a wrapper for ODE_Goodwin_Oscillator that
% allows for a vectorised input.
%
% Date: December 14, 2018

    [nObs, nDim] = size(Q);
    p = zeros(nObs, 1);
    D = zeros(nObs, nDim);
    for i = 1:nObs
        [p(i), D(i, :)] = ODE_Goodwin_Oscillator(Q(i, :), g);
    end
end
