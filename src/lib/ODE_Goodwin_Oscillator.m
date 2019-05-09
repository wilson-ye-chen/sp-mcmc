function [lpo,dlpo_dq] = ODE_Goodwin_Oscillator(q,g)
% The g-variable Goodwin Oscillator ODE
% Input q is a parameter vector, defined on the whole of R^d
% Here d = g + 2 where g is user-set
% All details are identical to in "The Controlled Thermodynamic Integral ..."
% The data-generating value is q = [0 1.1 0.69 0 ... 0 -0.69]
% Output is the log-posterior (lpo) density and its gradient (dlpo_dq)

%% Parameter Transformation
if size(q,1) == 1
    q = q'; % ensure q is a column vector
end
p = exp(q); % non-negativity constraint

%% Gradient of Transformation
dp_dq = exp(q);

%% Jacobean of Transformation
J = exp(sum(q));

%% Gradient of Log-Jacobean
dlJ_dq = ones(size(q));

%% Log-prior for p
lprp = sum(log(gampdf(p,2,1)));

%% Gradient of Log-prior for p
dlprp_dp = 1./p - 1;

%% ODE System
% g = 4; % number of variables in the Goodwin Oscillator
rho = 10; % exponent in the ODE
dimx = g; % number of state variables
dimp = g + 2; % number of parameters
x_init = zeros(1,g); % initial condition
% gradient field
f = @(t,x,p) [p(1)/(1 + p(2)*x(g)^rho) - p(g+2)*x(1); ...
              p(3:(g+1)).*x(1:(g-1)) - p(g+2)*x(2:g)];

%% Analytical Derivative Expressions
df_dx = @(t,x,p) [-p(g+2), zeros(1,g-2), -p(1)*p(2)*rho*x(g)^(rho-1)*(1+p(2)*x(g)^rho)^(-2); ...
                  [diag(p(3:(g+1))) zeros(g-1,1)] - [zeros(g-1,1) p(g+2)*eye(g-1)] ];
df_dp = @(t,x,p) [(1+p(2)*x(g)^rho)^(-1), -p(1)*x(g)^rho*(1+p(2)*x(g)^rho)^(-2), zeros(1,g-1), -x(1); ...
                  zeros(g-1,2), diag(x(1:(g-1))), -x(2:g) ];

%% Augmented Gradient Field
% augment the ode to include the sensitivity equations
% let z = [x1,...,x_dimx,dx1/dp1,...,dx_dimx/dp1,......,dx_dimx/dp_dimp]
f_aug = @(t,z,p) [f(t,z(1:dimx),p); ...
                  reshape(df_dx(t,z,p) * reshape(z((dimx+1):end),dimx,dimp) + df_dp(t,z,p),dimx*dimp,1)];

%% Dataset
rng(0); % one fixed dataset
p0 = [1; 3; 2; ones(g-2,1); 0.5]; % true parameter
z_init = [x_init, zeros(1,dimx*dimp)];
ts = 41:80; % measurement times
n = length(ts); % number of measurements
opt = odeset('RelTol',1e-2,'AbsTol',1e-5);
[~,z0] = ode45(@(t,z) f_aug(t,z,p0),ts,z_init,opt);
sigma = 0.1; % measurement standard deviation
y = z0(:,1:2) + sigma * randn(n,2); % only x1 and x2 observed
rng('shuffle'); % un-fix the random seed

%% Numerical Solution
[~,z] = ode45(@(t,z) f_aug(t,z,p),ts,z_init,opt);
% plot(ts,z0(:,1),ts,z0(:,2))

%% Log-likelihood for p
llp = -(n/2)*log(2*pi*sigma^2) - (2*sigma^2)^(-1) * norm(y-z(:,1:2),'fro')^2; % only x1 and x2 observed

%% Gradient of Log-likelihood for p
dllp_dp = zeros(dimp,1);
S = reshape(z0(:,(dimx+1):end),n,dimx,dimp);
for i = 1:dimp
    for j = 1:2 % only x1 and x2 observed
        dllp_dp(i) = dllp_dp(i) + sigma^(-2) * (y(:,j) - z(:,j))' * squeeze(S(:,j,i));
    end
end

%% Log-posterior for p
lpop = lprp + llp;

%% Log-posterior for q
lpo = lpop + log(J);

%% Gradient of Log-posterior for p
dlpop_dp = dlprp_dp + dllp_dp;

%% Gradient of Log-posterior for q
dlpo_dq = dlpop_dp .* dp_dq + dlJ_dq;

end
