function [xMin, dMin, k0Min, nEval] = fmin_grw_rand( ...
    f, X, G, fscr, fp, nIter, x0, S0, lambda, alpha, fileOut)
% [xMin, dMin, k0Min, nEval] = fmin_grw_rand(f, X, G, fscr, fp, nIter, ...
% x0, S0, lambda, alpha, fileOut) minimises the objective function via a
% random-walk Metropolis algorithm where the proposal distributions can be
% constructed adaptively based on the current points.
%
% Date: December 10, 2018

    [nObs, nDim] = size(X);
    iNew = nObs + 1;

    % Adapt proposal based on X
    if alpha(iNew) == 1
        S = S0;
    else
        S = alpha(iNew) .* S0 + (1 - alpha(iNew)) .* lambda .* cov(X);
        S = nearestSPD(S);
    end

    % If current point set is empty
    if nObs == 0
        start = x0;
    else
        start = X(randi(nObs), :);
    end

    % Run MCMC
    [Chn, acc] = grwmetrop(fp, start, S, nIter(iNew));

    % Evaluate vectorised objective
    C = unique(Chn(2:end, :), 'rows');
    D = fscr(C);
    [fVal, K0] = f(C, D);
    [~, iMin] = min(fVal);
    xMin = C(iMin, :);
    dMin = D(iMin, :);
    k0Min = K0(iMin, :);
    nEval = nIter(iNew) + size(C, 1);

    % Write output to disc
    if fileOut
        name = 'fmin_grw_rand_out.mat';
        if exist(name, 'file') == 2
            load(name);
            Chain = [Chain; Chn];
            acInd = [acInd; acc];
            obIdx = [obIdx; repmat(iNew, nIter(iNew), 1)];
            SttPt = [SttPt; start];
            MinPt = [MinPt; xMin];
            EvCnt = [EvCnt; nEval];
        else
            Chain = Chn;
            acInd = acc;
            obIdx = repmat(iNew, nIter(iNew), 1);
            SttPt = start;
            MinPt = xMin;
            EvCnt = nEval;
        end
        save(name, 'Chain', 'acInd', 'obIdx', 'SttPt', 'MinPt', 'EvCnt');
    end
end
