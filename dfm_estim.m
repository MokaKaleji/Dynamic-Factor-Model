%% DFM_estim.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Loadings and Transition Matrix with Dynamic Factors.
% University of Bologna
% Description:
%   Implements Quasi-Maximum Likelihood estimation of a Dynamic Factor Model
%   using the EM algorithm as per Barigozzi & Luciani (2021). This script
%   standardizes the data, invokes the BL_Estimate routine, and returns both
%   EM- and PCA-based estimates. 
%   Estimation code is directly from original author: Matteo Luciani.

clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset Selection, Frequency, Training Sample Size, and Standardization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Allow the user to select the dataset periodicity (monthly or quarterly), 
% load the corresponding data, specify the training sample size, and 
% standardize the data for numerical stability in model estimation.
% Explanation:
% The dynamic factor model requires a multivariate time series dataset. 
% This section provides a user-friendly interface to choose between pre-processed
% monthly or quarterly datasets, ensuring flexibility in periodicity. 
% The training sample size (T_train) is specified to focus on a subset of 
% the data, which is useful for in-sample estimation and out-of-sample 
% forecasting. Standardization (zero mean, unit variance) is applied to prevent
% numerical issues and ensure consistent scaling across variables, a common
% practice in high-dimensional time series modeling.

% --- Present Available Periodicity Options and Capture User Choice ---
% Purpose: Display a dialog for the user to select dataset periodicity.
% Explanation: The listdlg function provides a graphical interface to choose
% between monthly ('MD1959.xlsx') and quarterly ('QD1959.xlsx') datasets. 
% The selection is validated to ensure a choice is made, halting execution 
% if cancelled to prevent undefined behavior.
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg('PromptString','Select dataset:',...
                             'SelectionMode','single',...
                             'ListString',options,...
                             'Name','Dataset Selection',...
                             'ListSize',[400 200]);
if ~ok
    error('Dataset selection cancelled. Exiting script.');
end
% --- Load Data Based on Frequency ---
% Purpose: Load the selected dataset from an Excel file and extract the time
% series data.
% Explanation: The filepath is constructed based on the user's choice, 
% pointing to pre-processed datasets stored in a specific directory. The data
% is read into a table using readtable, then converted to a numeric array. 
% For quarterly data, the first column (date index) is excluded, as it is not
% part of the time series. The dimensions T (time points) and N (variables)
% are extracted for subsequent processing.
switch choiceIndex
    case 1                                                                 % Monthly frequency
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/MD1959.xlsx'];
        raw = readtable(filepath);
        x = table2array(raw);                                              % Include all series
        T = size(x,1);
    case 2                                                                 % Quarterly frequency
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/QD1959.xlsx'];
        raw = readtable(filepath);
        x = table2array(raw(:,2:end));                                     % Drop date index
        T = size(x,1);
    otherwise
        error('Unexpected selection index.');
end
[N_obs, N] = size(x);
% --- Define Training Sample Size ---
% Purpose: Prompt the user to specify the number of observations for the 
% training sample.
% Explanation: The training sample size (T_train) determines the subset of 
% data used for model estimation, allowing the remaining observations for 
% out-of-sample validation or forecasting. A default value of 225 is suggested,
% but the user can input any integer between 1 and T-1. The input is validated
% to ensure it is positive and less than the total number of observations, 
% preventing invalid training periods.
defaultTrain = '225';
prompt = sprintf(['Dataset has %d observations. Enter training size ' ...
    '(T_train):'], T);
userInput = inputdlg(prompt, 'Training Horizon', [3 100], {defaultTrain});
if isempty(userInput)
    error('No training size provided. Exiting.');
end
T_train = str2double(userInput{1});
assert(T_train>0 && T_train<T, 'T_train must be integer in (0, %d)', T);
% --- Standardization ---
% Purpose: Standardize the training data to zero mean and unit variance.
% Explanation: Standardization is critical for numerical stability in 
% high-dimensional factor models, as variables with different scales can lead
% to ill-conditioned matrices or biased factor estimates. The training data
% (first T_train observations) is centered by subtracting the mean and scaled
% by dividing by the standard deviation, computed across the training sample.
% This ensures all variables contribute equally to the factor structure and
% prevents numerical overflow in the EM algorithm.
x_train = x(1:T_train, :);
mean_train = mean(x_train);
std_train  = std(x_train);
x_train_norm = (x_train - mean_train) ./ std_train;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Running EM-based QML Dynamic Factor Model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call BL_Estimate with user-specified dimensions
%   r   = number of static factors
%   q   = number of common shocks
%   p   = VAR lag order on factors
%   iter= maximum EM iterations
%   tresh = convergence threshold for EM

r      = 6;    % Number of latent static factors
q      = 6;    % Number of common shock innovations
p      = 4;    % VAR lag order for factor dynamics
iter   = 100; % Maximum EM iterations
tresh  = 1e-4; % EM convergence tolerance

[EM, PCA] = BL_Estimate(x_train_norm, r, q, p, iter, tresh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save Results for Forecasting and Further Analysis 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM contains fields: A (factor loadings), Q, H, R, Kalman smoothing outputs, log-likelihood
% PCA contains principal component estimates for comparison
save('dfm_estim_results.mat', 'r', 'q', 'EM', 'PCA', ...
     'x_train', 'x_train_norm', 'mean_train', 'std_train', 'T_train', ...
     'p', 'N', 'iter', 'tresh');
disp('DFM estimation complete. Results saved.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference:
%   "Quasi Maximum Likelihood Estimation and Inference of Large Approximate
%    Dynamic Factor Models via the EM algorithm"
%   Barigozzi & Luciani (2021), arXiv:1910.03821vXXX
% 
% Usage:
%   [EM, PCA] = BL_Estimate(x, r, q, p, iter, tresh, s01)
% Inputs:
%   x      - standardized data matrix 
%   q      - number of common shocks
%   p      - VAR lag order on factors
%   iter   - EM maximum iterations
%   tresh  - convergence threshold for EM
%   s01    - flag to standardize data internally
% Outputs:
%   EM     - structure with EM-based QML estimates
%   PCA    - structure with initial PCA-based estimates
% 
% Original author: Matteo Luciani (matteoluciani@yahoo.it)
% Function signature and implementation details are unchanged to preserve
% methodological fidelity.


function [EM, PCA]=BL_Estimate(x,r,q,p,iter,tresh)


[T, N] = size(x);
pr=p*r;

                                                                        
xx=x; mx=zeros(1,N); sx=ones(1,N);                                     
                                                                        
MX=repmat(mx,T,1); SX=repmat(sx,T,1);                                       


%%% ======================================== %%%
%%%  Estimate factors and loadings with PCA  %%%
%%% ======================================== %%%
Gamma=cov(xx);                                                              % covariance matrix of the standardize data
[W,M] = eigs(Gamma, r,'LM'); W=W*diag(sign(W(1,:))');                       % eigenvalue-eigenvectors
L0=W*sqrt(M);                                                               % estimate factor loadings
f0=xx*W/sqrt(M);                                                            % estimate common factors
e0=xx-f0*L0';                                                               % "residual" PCA
[~, uu0,AL0]=ML_VAR(f0,p,0);                                                % VAR on the Static Factors
[~, G0]=ML_edynfactors2(uu0,q);                                             % Common Shocks
chi0=MX+(f0*L0').*SX;                                                       % common component PCA
xi0=x-chi0;                                                                 % idiosyncratic component PCA

%%% ==================================== %%%
%%%  Expectation Maximization Algorithm  %%%
%%% ==================================== %%%
AL0b=cat(3,AL0,zeros(r,r));                                                 % add one lag to avoid computing PtTm
[F0,lambda0,A0,P0,Q0,R0]=ML_SS_DFM_I0(f0,L0,AL0b,G0,e0,N,p+1,r);            % State-Space representation
[xitT,PtT1,~,~,~,~,A1,L1,R1,Q1,G1]=...                                      % EM algorithm
    ML_efactorML4c3(xx,F0,P0,A0,lambda0,R0,Q0,r,q,p,iter,tresh,0);          % ------------
PtT1=PtT1(1:pr,1:pr,:);A1=A1(1:pr,1:pr); Q1=Q1(1:pr,1:pr);                  % Eliminate the extra lag
f1=xitT(:,1:r); L1=L1(:,1:pr);                                              % factors
jr=diag(sign(L1(1,1:r))'); L1(:,1:r)=L1(:,1:r)*jr; f1=f1*jr;                % Loadings of first variable are positive (normalization)
chi1=MX+(f1*L1(:,1:r)').*SX;                                                % Common component
xi1=x-chi1;                                                                 % idiosyncratic component

%%% ==================== %%%
%%%  Saves coefficients  %%%
%%% ==================== %%%
PCA.F=f0;
PCA.Lambda=L0;
PCA.chi=chi0;
PCA.xi=xi0;
PCA.R=R0;
PCA.G=G0;
PCA.A=A0;
PCA.Q=Q0;
PCA.P=P0;

EM.F=f1;
EM.Lambda=L1;
EM.chi=chi1;
EM.xi=xi1;
EM.R=R1;
EM.G=G1;
EM.A=A1;
EM.Q=Q1;
EM.PtT=PtT1;
EM.xitT=xitT;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ML_SS_DFM_I0 - Build State-Space representation of stationary DFM
% 

function [F0,lambda0,A0,P0,Q0,R0]=ML_SS_DFM_I0(f0,L0,AL0,G0,xi0,N,p,r,IDEN)

if nargin <9; IDEN=1; end 
pr=p*r;
Q0 = zeros(pr,pr); Q0(1:r,1:r)=G0*G0';                                      % Variance of the VAR residuals
lambda0 = [L0 zeros(N,r*(p-1))];                                            % Loadings
R0 = diag(diag(cov(xi0)));                                                  % covariance idio
F0=[]; for pp=1:p; F0=cat(1,F0,f0(p+1-pp,:)'); end                          % initial conditions factors
A0=ML_VAR_companion_matrix(AL0);                                            % VAR companion form aka VAR(1)

if IDEN==1; P0=eye(pr);                                                     % Impose initial variance = identity matrix
else; P0 = reshape(inv(eye(pr^2)-kron(A0,A0))*Q0(:),pr,pr);                 % initial variance
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_efactorML4c - ML estimation I0 DFM, straight EM no initialization, P00=I (for simulations)
% 
% [xitT,PtT,PtTm,xitt,xittm,Ptt,Pttm,A,Lambda,R,Q]=...
%     ML_efactorML4(y,F0,P0,A,Lambda,R,Q,q,p,maxiter,tresh,cc)% 
% 
%       Y - data with deterministic component
%      F0 - initial values for the states
%      P0 - initial variance for the states
%       A - VAR(1) for the states
%  Lambda - Loadings
%       R - covariance matrix errors
%       Q - covariance matrix shocks
%       r - number of factors
%       q - number of shocks
%       p - number of lags VAR factors
% maxiter - max number iterations
%   tresh - threshold for stopping algorithm
%      cc - # of obs eliminated before estimating parameters
% 


function [xitT,PtT,xitt,xittm,Ptt,Pttm,A,Lambda,R,Q,G]=...
    ML_efactorML4c3(y,F0,P0,A,Lambda,R,Q,r,q,p,maxiter,tresh,cc)


OPTS.disp = 0;
[T, N]=size(y);                                                             % size of the panel
pr=p*r; pr1=(p+1)*r;

%%% =================== %%%
%%%  Start with E-Step  %%%
%%% =================== %%%
[xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter2(F0,P0,y,A,Lambda,R,Q);        % Kalman Filter
[xitT,PtT]=ML_KalmanSmoother3(y,A,xitt',xittm',Ptt,Pttm,Lambda,R);          % Kalman Smoother
F00=xitT(1,:)'; P00=eye(pr1);                                               % initial conditions  
F=xitT(:,1:r);                                                              % factors
loglik1=loglik;                                                             % likelihood

for jj=1:maxiter         
    
    %%%% ======================== %%%%
    %%%% ====                ==== %%%%
    %%%% ====    M - Step    ==== %%%%
    %%%% ====                ==== %%%%
    %%%% ======================== %%%%
    
    %%% =========================== %%%
    %%% 1: Estimate factor Loadings %%%
    %%% =========================== %%%
    yy=ML_center(y)';                                                       % endogenous variables,     
    xx=ML_center(F)';                                                       % exogenous variables, aka the factors        
    num=zeros(N,r); den=zeros(r,r);                                         % preallocates
    for tt=cc+1:T  % ------------------------------------------------------ % \sum_{t=1}^T
        num=num+yy(:,tt)*xx(:,tt)';                                         % y_t F_{t|T}'
        den=den+xx(:,tt)*xx(:,tt)'+PtT(1:r,1:r,tt);                         % F_{t|T}F_{t|T}'+P_{t|T}
    end    
    L=num/den;                                                              % factor loadings           
    Lambda(1:N,1:r) = L;                                                    % store factor loadings

    %%% ======================================================= %%%
    %%% 2: Estimate parameters for law of motion of the Factors %%%
    %%% ======================================================= %%%  
    yy=cat(2,zeros(r,1),F(2:T,:)');                                         % F_t
    xx=cat(2,ML_lag(xitT(:,1:pr),1)',zeros(pr,1));                          % F_{t-1}
    EF1=zeros(r,pr); EF1F1=zeros(pr,pr); EF=zeros(r,r);                     % initialize
    for tt=cc+2:T  % ------------------------------------------------------ % \sum_{t=1}^T
        EF1=EF1+yy(:,tt)*xx(:,tt-1)'+PtT(1:r,r+1:pr1,tt);                   % E(F_t F_{t-1})
        EF1F1=EF1F1+xx(:,tt-1)*xx(:,tt-1)'+PtT(1:pr,1:pr,tt-1);             % E(F_{t-1} F_{t-1})
        EF=EF+yy(:,tt)*yy(:,tt)'+PtT(1:r,1:r,tt);                           % E(F_t F_t)
    end   
    
    A(1:r,1:pr)=EF1/EF1F1;                                                  % parameter VAR factors        
    Q(1:r,1:r) = (EF - A(1:r,1:pr)*EF1') / (T-cc);                          % covariance
    if r>q; [P,M] = eigs(Q(1:r,1:r),q,'lm',OPTS); Q(1:r,1:r) = P*M*P'; end  % in case the factors are singular
    
    %%% ============================== %%%
    %%% 3: Covariance Prediction Error %%%
    %%% ============================== %%%
    yy=y';
    xx=xitT';
    R=zeros(N,N);                                                           % preallocates
    for tt=cc+1:T  % ------------------------------------------------------ % \sum_{t=1}^T
        eta=yy(:,tt)-Lambda*xx(:,tt);                                       % prediction error
        R = R+ eta*eta'+ Lambda*PtT(:,:,tt)*Lambda';                        % E(eta_t eta_t')
    end
    R = diag(diag(R/(T-cc)));
    
    %%%% ======================== %%%%
    %%%% ====                ==== %%%%
    %%%% ====    E - Step    ==== %%%%
    %%%% ====                ==== %%%%
    %%%% ======================== %%%%            
    [xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter2(F00,P00,y,A,Lambda,R,Q);  % Kalman Filter
    [xitT,PtT]=ML_KalmanSmoother3(y,A,xitt',xittm',Ptt,Pttm,Lambda,R);      % Kalman SMoother
    F=xitT(:,1:r);                                                          % jj-th step factors            
    F00=xitT(1,:)'; P00=eye(pr1);                                           % initial conditions    

    %%%% ================================= %%%%
    %%%% ====                         ==== %%%%
    %%%% ====    Check convergence    ==== %%%%
    %%%% ====                         ==== %%%%
    %%%% ================================= %%%%
fprintf('EM iter %3d: loglik = %.6e\n', jj, loglik);

if jj > 1
    % 2) Compute relative change
    delta     = abs(loglik - loglik1);
    avgLik    = (abs(loglik) + abs(loglik1) + 1e-3)/2;
    relChange = delta / avgLik;
    
    % 3) Print the relative change
    fprintf(' Loglik rel change at iter %d: % .2e\n', jj, relChange);
        
    % 4) Stop if we’ve converged
    if relChange < tresh
        fprintf('Converged at iteration %d (rel change = %.2e < %.1e)\n', ...
            jj, relChange, tresh);
        break;
    end
end

% 6) Prepare for next iteration
loglik1 = loglik;
end

if r>q; G = P*sqrt(M); else; G=eye(r); end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_KalmanFilter2 - Kalman Filter Estimation of a Factor Model
% 
%  THE MODEL
% 
%   X_t = beta1 + beta2*t + C S_t + e_t,        e_t ~ N(0,R), R is diagonal
%   S_t = mu + A S_{t-1} + u_t,                 u_t ~ N(0,Q)
% 
%       S_t|X^{t-1} ~ N( S_{t|t-1} , P_{t|t-1} )
%       X_t|X^{t-1} ~ N( X_{t|t-1} , H_{t|t-1} )
% 
% 
%  THE PROCEDURE
% 
% [xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter(initx,initV,x,A,C,R,Q,mu,beta)
% 
% INPUTS:
%   x - the data
%   C - the observation matrix 
%   A - the system matrix
%   Q - the system covariance 
%   R - the observation covariance
%   initx - the initial state (column) vector 
%   initV - the initial state covariance 
%   mu   - constant in transition equation (optional)
%   beta - constant and linear trend in observation equation (optional)
% OUTPUTS:
%   xittm = S_{t|t-1}
%    Pttm = P_{t|t-1}
%    xitt = S_{t|t} 
%     Ptt = P_{t|t}
%  loglik = value of the log-likelihood
% 
% Matteo Luciani (matteoluciani@yahoo.it)

function [xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter2(initx,initV,x,A,C,R,Q,mu,beta)

N=size(x,2);
T=size(x,1);                                                                % Number of Observations
r=size(A,1);                                                                % Number of states
xittm=[initx zeros(r,T)];                                                   % S_{t|t-1}
xitt=zeros(r,T);                                                            % S_{t|t} 
Pttm=cat(3,initV,zeros(r,r,T));                                             % P_{t|t-1}
Ptt=zeros(r,r,T);                                                           % P_{t|t}
loglik=0;                                                                   % Initialize the log-likelihood
y=x';                                                                       % transpose for convenience

if nargin<8; mu=zeros(r,1); end
if nargin<9; beta=zeros(1,2); end

for j=1:T
    
    %%% ============= %%%
    %%% Updating Step %%%
    %%% ============= %%%   
    X=C*xittm(:,j) + beta*[1;j];                                            % X_{t|t-1} - Prediction
    H=C*Pttm(:,:,j)*C'+R;                                                   % H_{t|t-1} - Conditional Variance of the Observable   
    Hinv = inv(H);    
    e = y(:,j) - X;                                                         % error (innovation)
    xitt(:,j)=xittm(:,j)+Pttm(:,:,j)*C'*Hinv*e;                             % S_{t|t}
    Ptt(:,:,j)=Pttm(:,:,j)-Pttm(:,:,j)*C'*Hinv*C*Pttm(:,:,j);               % P_{t|t}   

    %%% =============== %%%
    %%% Prediction Step %%%
    %%% =============== %%%
    xittm(:,j+1)=mu+A*xitt(:,j);                                            % S_{t|t-1} - States
    Pttm(:,:,j+1)=A*Ptt(:,:,j)*A'+Q;                                        % P_{t|t-1} - Conditional Variance of the State 
    loglik = loglik + 0.5*(-N * log(2*pi) - log(det(Hinv))  - e'*Hinv*e);                    % Log Likelihood   
    
end

xitt=xitt';
xittm=xittm';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_KalmanSmoother3 - Kalman Smoother for a Factor Model
% 
%  THE MODEL
%  
%   X_t = C S_t + e_t,              e_t ~ N(0,R), R is diagonal
%   S_t = A S_{t-1} + u_t,          u_t ~ N(0,Q)
% 
%   S_t|X^{t-1} ~ N( S_{t|t-1} , P_{t|t-1} )
%   X_t|X^{t-1} ~ N( X_{t|t-1} , H_{t|t-1} )
% 
%  
%  THE PROCEDURE
%  
% [xitT,PtT,PtTm]=ML_KalmanSmoother3(x,A,xitt,xittm,Ptt,Pttm,C,R)
% 
% INPUTS:
%       A - the system matrix
%    xitt - S_{t|t}
%   xittm - S_{t|t-1}
%     Ptt - P_{t|t}
%    Pttm - P_{t,t-1|t-1}
%       C - the observation matrix 
%       R - the observation covariance
% OUTPUTS:
%    xitT = S_{t|T}
%     PtT = P_{t|T} 
%    PtTm = P_{t,t-1|T} 
%     
% This use the formulas in Chapter 4 of Durbin and Koopman "Time Series Analysis by
% State Space Methods" (second edition)
% Matteo Luciani (matteoluciani@yahoo.it)

function [xitT,PtT]=ML_KalmanSmoother3(x,A,xitt,xittm,Ptt,Pttm,C,R)

[T]=size(xitt,2);                                                           % Number of Observations
[n, r]=size(C);                                                             % Number of Variables and Number of States
Pttm=cat(3,zeros(r,r,1),Pttm(:,:,1:end-1));                                 % P_{t|t-1}, remove the last observation because it has dimension T+1
xittm=[zeros(r,1) xittm(:,1:end-1)];                                        % S_{t|t-1}, remove the last observation because it has dimension T+1
xitT=[zeros(r,T)  xitt(:,T)];                                               % S_{t|T} 
PtT=cat(3,zeros(r,r,T),Ptt(:,:,T));                                         % P_{t|T} 
y=[zeros(n,1) x'];                                                          % transpose for convenience

rr=zeros(r,T+1); N=zeros(r,r,T+1); L=zeros(r,r,T+1);                        % Preallocates

CPCR = C*Pttm(:,:,T+1)*C'+R;                                                % useful matrices
iCPCR = eye(n)/CPCR;                                                        % ---------------
CiCPCRC = C'*iCPCR *C;                                                      % ---------------
L(:,:,T+1) = A - A*Pttm(:,:,T+1)*CiCPCRC;                                   % ---------------

for tt = T:-1:2    
    CPCR = C*Pttm(:,:,tt)*C'+R;                                             % useful matrices
    iCPCR = eye(n)/CPCR;                                                    % ---------------
    CiCPCRC = C'*iCPCR *C;                                                  % ---------------    
    rr(:,tt-1) = C'*iCPCR * ( y(:,tt) - C*xittm(:,tt) ) + L(:,:,tt)'*rr(:,tt);    
    N(:,:,tt-1) = CiCPCRC + L(:,:,tt)'*N(:,:,tt)*L(:,:,tt);    
    L(:,:,tt) = A - A*Pttm(:,:,tt)*CiCPCRC;             
    xitT(:,tt) = xittm(:,tt) + Pttm(:,:,tt)*rr(:,tt-1);                     % state        
    PtT(:,:,tt) = Pttm(:,:,tt) - Pttm(:,:,tt)*N(:,:,tt-1)*Pttm(:,:,tt);     % covariance of the state 
end

PtT(:,:,1)=[];
xitT=xitT(:,2:end)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_center - Demean variables
% CENTER XC = center(X)
%	Centers each column of X.
%	J. Rodrigues 26/IV/97, jrodrig@ulb.ac.be
function XC = ML_center(X)
T = size(X,1);
XC = X - ones(T,1)*(sum(X)/T); % Much faster than MEAN with a FOR loop
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_lag - Given x(t) produces x(t-j0) .... x(t-k)
%
% xx=ML_lag(x,k,j0)
% by default j0 is 1
%
% eg x=[x1 x2]
% xx=ML_lag(x,2)=[x1_{t-1} x1_{t-2} x2_{t-1} x2_{t-2}]
%
% Matteo Luciani (matteoluciani@yahoo.it)

function xx=ML_lag(x,k,j0)
[T, N] = size(x);

if nargin<3; j0=1; end;    
n=1;

if N*(k+1-j0)==0
    xx=[];
elseif T==1
    xx=x;
else
    xx=zeros(T-k,N*(k+1-j0));
    for i=1:N
        for j=j0:k
            xx(:,n)=x(k+1-j:T-j,i);
            n=n+1;
        end;
    end;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_VAR - Estimates a VAR(k) for a vector of variables y
% if y is a single variable it estimates an AR model with OLS
%
% [A,u,AL,CL]=ML_VAR(y,det,jj);
%   y  = vector of endogenous variables
%   k  = number of lags
%   det = 0 no constant
%   det = 1 constant
%   det = 2 time trend
%   det = 3 constant + time trend
%   A  = Matrix of coefficients for the reduced form 
%   u  = Vector of Residuals
%
%  y_t=A(L)y_{t-1}+u_t
%  y_t=C(L)u_t
% 
% written by Matteo Luciani (matteoluciani@yahoo.it)

function [A,u,AL,CL,C1]=ML_VAR(y,k,det,s)
[T, N] = size(y);

%%% Building Up the vector for OLS regression %%%
yy=y(k+1:T,:);
xx=NaN(T-k,N*k);
for ii=1:N    
    for jj=1:k
        xx(:,k*(ii-1)+jj)=y(k+1-jj:T-jj,ii);
    end
end

%%% OLS Equation-By-Equation %%%
if det==0; ll=0; elseif  det==3; ll=2; else; ll=1; end
A=NaN(N*k+ll,N); u=NaN*yy;
for ii=1:N
    [A(:,ii),u(:,ii)]=ML_ols(yy(:,ii),xx,det);
end

At=A; if det==3; At(1:2,:)=[]; elseif det==1||det==2; At(1,:)=[]; end
AL=NaN(N,N,k); for kk=1:k; AL(:,:,kk)=At(1+kk-1:k:end,:)'; end  


%%% Impulse Responses %%%
if nargin<4;s=20; end
CL(:,:,1) = eye(N);
for ss=2:s
    CL(:,:,ss) = 0;
    for ii = 1:min(ss-1,k)        
        temp3=AL(:,:,ii)*CL(:,:,ss-ii);        
        CL(:,:,ss)=CL(:,:,ss)+temp3;
    end
end

C1=eye(N); for ii=1:k; C1=C1-AL(:,:,ii); end; C1=inv(C1);                     % Long Run Multipliers
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_ols - OLS Estimation
% 
% [beta,u,v,esu,r2,espqr]=ML_ols(y,x,det);
%   Inputs:
%       y   = Endogenous Variable (vector)
%       x   = Exogenous Variables (matrix)
%       det = 0 no constant
%       det = 1 constant
%       det = 2 time trend
%       det = 3 constant + time trend
%   Outputs:
%       beta  = estimated coefficient
%       u     = residuals
%       v     = varcov matrix of the estimates
%       esu   = Residual Variance
%       r2    = R-Squared
%       espqr = estimates standard errors
%
% Written by Matteo Luciani (matteoluciani@yahoo.it)
% This is a modified version of the codes available on Fabio Canova webpage

function [beta,u,v,esu,r2,espar,yhat]=ML_ols(y,x,det)
T = size(x,1);

cons=ones(T,1); trend=(1:1:T)';
if      det==1; x=[cons x];
elseif  det==2; x=[trend x];
elseif  det==3; x=[cons trend x];
end;
k=size(x,2);                                                                % number of parameters
xx=eye(k)/(x'*x);                                                           % inv(x'x)
beta=xx*x'*y;                                                               % ols coeff
yhat=x*beta;                                                                % fitted values
u=y-yhat;                                                                   % residuals
uu=u'*u;                                                                    % SSR
esu=uu/(T-k);                                                               % Residual Variance
yc=y-mean(y);                                                               % Centered variables
r2=1-(uu)/(yc'*yc);                                                         % R2
v=esu*xx;                                                                   % varcov matrix of the estimates
espar=sqrt(diag(v));                                                        % Standard errors
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_edynfactors2 - Estimation of Dynamic Factors Innovation 
% 
% Principal Components of the Residuals of a VAR(p) estimated on r Static Factors
% Refined version of ML_edynfactors in which svd rather than eigs is used
% [etahat, G]=ML_edynfactors2(y,q)
% The Model:
%       X(t) = lambda*F(t) + xsi(t)
%       F(t) = A(L)*F(t-1) + epsilon
%       epsilon = G*eta
%       where G is r by q
% Outputs:
%       etahat = estimates of dynamic Factors Innovations
%       G = estimates of G
% Inputs:
%       y = epsilon
%       q = number of dynamic factors
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function [eta, G]=ML_edynfactors2(y,q)
opt.disp=0;
N=size(y,2);
sigma=cov(y);                                                               % Variance Covariance Matrix of VAR Residuals
if q<N; [P,M]=eigs(sigma,q,'LM',opt); else [P,M]=eig(sigma); end            % Eigenvalue eigenvectors decomposition
eta=y*P*(M^-.5);                                                            % Dynamic Factor Innovations            
G=P(:,1:q)*(M^.5);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML_VAR_companion_matrix - build the companion matrix for a VAR
% PHI=ML_VAR_companion_matrix(AL,det);
% Y(t)=AL(:,:,1)*Y(t-1)+...+AL(:,:,p)*Y(t-p)+u(t)
% For a VAR(3) with k variables:
% | AL(:,:,1)   AL(:,:,2)   AL(:,:,3)   |
% | eye(k,k)    zeros(k,k)  zeros(k,k) |
% | zeros(k,k)  eye(k,k)    zeros(k,k) |
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function PHI=ML_VAR_companion_matrix(A)

s=size(A);
if length(s)==1; PHI=A;                             % AR(1)
elseif length(s)==2;                        
    if s(2)==1; p=s(1,1);                           % Autoregressive model
        if  p==2; PHI=[A'; 1 0];                    % AR(2)   
        else PHI=diag(ones(1,p-1),-1); PHI(1,:)=A'; % AR(p)
        end
    else PHI=A; end                                 % VAR(1)
else    
    k=s(1,1); p=s(1,3);
    PHI=[];
    for i=1:p; PHI=[PHI A(:,:,i)]; end;
    PHI=[PHI;eye(k*(p-1),k*(p-1)) zeros(k*(p-1),k)];
end 
end
