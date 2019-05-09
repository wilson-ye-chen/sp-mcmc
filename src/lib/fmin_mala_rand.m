function [xMin, dMin, k0Min, nEval] = fmin_mala_rand( ...
    f, X, G, fscr, fp, nIter, x0, S0, h, alpha, fileOut)
% [xMin, dMin, k0Min, nEval] = fmin_mala_rand(f, X, G, fscr, fp, nIter, ...
% x0, S0, h, alpha, fileOut) minimises the objective function via a Metropolis
% -adjusted Langevin algorithm where the proposal distributions can be made
% adaptive based on the current points.
%
% Date: February 1, 2019

    [nObs, nDim] = size(X);
    iNew = nObs + 1;

    % Adapt proposal based on X
    if alpha(iNew) == 1
        S = S0;
    else
        S = alpha(iNew) .* S0 + (1 - alpha(iNew)) .* cov(X);
        S = nearestSPD(S);
    end

    % If current point set is empty
    if nObs == 0
        start = x0;
    else
        start = X(randi(nObs), :);
    end

    % Run MCMC
    [Chn, D, ~, acc] = mala(fp, fscr, start, h, S, nIter(iNew));

    % Evaluate vectorised objective
    [C, iUnq] = unique(Chn(2:end, :), 'rows');
    D = D(iUnq, :);
    [fVal, K0] = f(C, D);
    [~, iMin] = min(fVal);
    xMin = C(iMin, :);
    dMin = D(iMin, :);
    k0Min = K0(iMin, :);
    nEval = 2 .* nIter(iNew);

    % Write output to disc
    if fileOut
        name = 'fmin_mala_rand_out.mat';
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
