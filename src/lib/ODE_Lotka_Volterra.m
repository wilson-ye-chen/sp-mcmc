function [lpo,dlpo_dq] = ODE_Lotka_Volterra(q)
% The Lotka-Volterra ODE
% Input q is a parameter vector, defined on the whole of R^3
% The data-generating value is q = [-1.61 -1.61 1.10]
% Output is the log-posterior (lpo) density and its gradient (dlpo_dq)

%% Parameter Transformation
if size(q,1) == 1
    q = q'; % ensure q is a column vector
end
p = exp(q); % non-negativity constraint

%% Gradient of Transformation
dp_dq = exp(q);

%% Jacobean of Transformation ###
J = exp(sum(q));

%% Gradient of Log-Jacobean ###
dlJ_dq = ones(size(q));

%% Log-prior for p ###
lprp = sum(log(normpdf(p))); 

%% Gradient of Log-prior for p ###
dlprp_dp = - p;

%% ODE System
dimx = 2; % number of state variables
dimp = 3; % number of parameters
x_init = [-1 1]; % initial condition
% gradient field
f = @(t,x,p) [p(3)*x(1) - (1/3)*p(3)*x(1)^3 + p(3)*x(2); ...
              -(1/p(3))*x(1) + (p(1)/p(3)) - (p(2)/p(3))*x(2)];

%% Analytical Derivative Expressions
df_dx = @(t,x,p) [p(3) - p(3)*x(1)^2, p(3); ...
                  -(1/p(3)),          -(p(2)/p(3))];
df_dp = @(t,x,p) [0, 0, x(1) - (1/3)*x(1)^3 + x(2); ...
                  (1/p(3)), -x(2)/p(3), (x(1)-p(1)+p(2)*x(2))/(p(3)^2)];       
             
%% Augmented Gradient Field
% augment the ode to include the sensitivity equations
% let z = [x1,...,x_dimx,dx1/dp1,...,dx_dimx/dp1,......,dx_dimx/dp_dimp]
f_aug = @(t,z,p) [f(t,z(1:dimx),p); ...
                  reshape(df_dx(t,z,p) * reshape(z((dimx+1):end),dimx,dimp) + df_dp(t,z,p),dimx*dimp,1)];
              
%% Dataset
rng(0); % one fixed dataset
p0 = [0.2; 0.2; 3]; % true parameter 
z_init = [x_init, zeros(1,dimx*dimp)];
ts = 0:20; % measurement times
n = length(ts); % number of measurements
[~,z0] = ode45(@(t,z) f_aug(t,z,p0),ts,z_init); 
sigma = 0.1; % measurement standard deviation   
y = z0(:,1:dimx) + sigma * randn(n,dimx);
rng('shuffle'); % un-fix the random seed              
              
%% Numerical Solution
[~,z] = ode45(@(t,z) f_aug(t,z,p),ts,z_init); 
%plot(ts,z0(:,1),ts,z0(:,2))

%% Log-likelihood for p
llp = -(n/2)*log(2*pi*sigma^2) - (2*sigma^2)^(-1) * norm(y-z(:,1:dimx),'fro')^2;

%% Gradient of Log-likelihood for p
dllp_dp = zeros(dimp,1);
S = reshape(z0(:,(dimx+1):end),n,dimx,dimp);
for i = 1:dimp
    for j = 1:dimx
        dllp_dp(i) = dllp_dp(i) + sigma^(-2) * (y(:,j) - z(:,j))' * squeeze(S(:,j,i));
    end
end

%% Log-posterior for p ###
lpop = lprp + llp;

%% Log-posterior for q ###
lpo = lpop + log(J);

%% Gradient of Log-posterior for p ###
dlpop_dp = dlprp_dp + dllp_dp;

%% Gradient of Log-posterior for q ###
dlpo_dq = dlpop_dp .* dp_dq + dlJ_dq;
                    
end



                    