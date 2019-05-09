function E = fstp_adagrad(i, PhiNew, master, mom)
% E = fstp_adagrad(i, PhiNew, master, mom) updates the step-size using
% the AdaGrad with momentum strategy.
%
% Input:
% i      - iteration index.
% PhiNew - nPar-by-nDim matrix of perturbation directions.
% master - master step-size, e.g., 0.1.
% mom    - momentum coefficient between 0 and 1, e.g., 0.9.
%
% Output:
% E      - nPar-by-nDim matrix of updated step-sizes.
%
% Date: May 13, 2017

    persistent Phi;
    fudge = 1e-6;
    if i <= 1
        Phi = PhiNew .^ 2;
    else
        Phi = mom .* Phi + (1 - mom) .* PhiNew .^ 2;
    end
    E = master ./ (fudge + sqrt(Phi));
end
