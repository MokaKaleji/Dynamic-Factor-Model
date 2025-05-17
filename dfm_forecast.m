%% DFM_forecasting.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Loadings and Transition Matrix with Dynamic Factors.
% University of Bologna
% Description:
%   Evaluates out-of-sample forecast accuracy for the QML Dynamic Factor Model
%   estimates generated via BL_Estimate. Computes mean squared forecast errors,
%   and visualizes forecast performance alongside latent factor dynamics.


clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset Selection, Load Data and Define Key Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Allow user to select a dataset (monthly or quarterly) and load 
% corresponding data.
% Explanation: Presents a dialog for selecting between monthly (MD1959.xlsx)
% or quarterly (QD1959.xlsx) datasets, loads the chosen data, and defines 
% key variables (GDP, Unemployment, Inflation, 1-Year Treasury - 3-Month Treasury)
% for forecasting evaluation. The datasets are high-dimensional time series
% used in macroeconomic forecasting, with specific indices for key variables.
% References:
%   - McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for 
%     macroeconomic research. Journal of Business & Economic Statistics, 
%     34(4), 574-589. (For FRED-MD dataset).
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg( ...
    'PromptString', 'Select the dataset to load:', ...
    'SelectionMode', 'single', ...
    'ListString', options, ...
    'Name', 'Dataset Selection', ...
    'ListSize', [400 200]);
if ~ok
    error('No dataset selected. Exiting...');
end
% Load data based on user selection
% Explanation: Loads the dataset from a specified filepath, sets the time 
% series length T, and extracts key variables’ indices. The monthly dataset
% has T=790 observations, and the quarterly has T=264, reflecting typical 
% macroeconomic data frequencies.
switch choiceIndex
    case 1
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/MD1959.xlsx'];
        T = 790;
        tableData = readtable(filepath);
        x = table2array(tableData);
        key_vars = [1, 24, 105, 89];                                       % Indices for key variables
    case 2
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/QD1959.xlsx'];
        T = 264;
        tableData = readtable(filepath);
        x = table2array(tableData(:,2:end));                               % Exclude ate column
        key_vars = [1, 58, 116, 147];                                      % Indices for key variables
    otherwise
        error('Unexpected selection index.');
end
var_names = {'GDP', 'Unemployment', 'Inflation', '1_Y Treasury - 3_M Treasury'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load DFTL Estimation Outputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('dfm_estim_results.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Forecast Horizon 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Prompt user to specify the forecast horizon H.
% Explanation: Computes the maximum possible horizon s = T - T_train 
% (test sample size) and validates that H is a positive integer between 
% 1 and s. This ensures forecasts are feasible within the test data range.
s = T - T_train;
H = input(sprintf('Enter forecast horizon H from 1 to T = %d: ', s));
if ~(isscalar(H) && isnumeric(H) && H == floor(H) && H >= 1 && H <= s)
    error('H must be an **integer** between 1 and T = %d.', s);            % Forecast horizon
end
disp(['Using forecast horizon H = ', num2str(H)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare Test Sample 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Extract and normalize the test data for forecasting evaluation.
% Explanation: Selects the test data x_test for the specified horizon H 
% from the full dataset and normalizes it using training data statistics 
% (mean_train, std_train) to match the scale of the estimation inputs,
% ensuring consistency.
x_test = x(T_train+1:T_train+H, :);    % Observations for evaluation
x_test_norm = (x_test - mean_train) ./ std_train;  % Normalize as in training

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Running Forecast Via DFTL Outputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[EM_Forecast] = DFM_forecast(EM, H, mean_train, std_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSFE/Ratio, RMSE, Diebold–Mariano Tests, Ljung–Box Tests, Encompassing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Evaluate forecast accuracy and compare against benchmarks.
% Explanation: Computes Mean Squared Forecast Error (MSFE), Root Mean 
% Squared Error (RMSE), MSFE ratios against random walk (RW) and AR(1) 
% benchmarks, and conducts statistical tests to assess forecast performance
% and residual properties.
% References:
%   - Diebold & Mariano (1995) for comparing predictive accuracy.
%   - Harvey & Todd (1983) for forecast encompassing tests.
%   - Ljung & Box (1978) for residual autocorrelation tests.

% --- MSFE and RMSE ---
% Purpose: Calculate MSFE and RMSE for key variables.
% Explanation: MSFE is the average squared forecast error, computed per 
% horizon and variable. RMSE is the square root of MSFE, providing a 
% scale-interpretable metric.
squared_errors = (x_test(:, key_vars) - EM_Forecast.yhat(:, key_vars)).^2;
MSFE_horizon = mean(squared_errors, 2);
MSFE_all = mean(squared_errors, 1);
MSFE_overall = mean(MSFE_all);

fprintf('Overall MSFE (original): %.6f\n', MSFE_overall);
for k = 1:length(key_vars)
    fprintf('MSFE (original) %s: %.6f\n', var_names{k}, MSFE_all(k));
end

RMSE = sqrt(MSFE_all);
for k = 1:length(key_vars)
    fprintf('RMSE %s: %.4f\n', var_names{k}, RMSE(k));
end

% --- Random Walk Benchmark ---
% Purpose: Compute MSFE for a naive random walk forecast.
% Explanation: The random walk forecast assumes y_{t+h} = y_t, using the 
% last training observation. MSFE ratios (DF vs. RW) indicate relative 
% performance.
yhat_rw = repmat(x(T_train, :), H, 1);
squared_errors_rw = (x_test(:, key_vars) - yhat_rw(:, key_vars)).^2;
MSFE_rw_all = mean(squared_errors_rw, 1);
MSFE_ratio_rw = MSFE_all ./ MSFE_rw_all;

% --- AR(1) Benchmark ---
% Purpose: Compute MSFE for an AR(1) forecast.
% Explanation: Fits an AR(1) model to each key variable’s training data and
% forecasts H steps ahead. MSFE ratios (DF vs. AR(1)) assess DFTL’s 
% performance against a simple time series model.
% Reference: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2008). Time 
% Series Analysis: Forecasting and Control. Wiley. (For ARIMA modeling).
clear forecast
ar1_errors = zeros(H, length(key_vars));
for k = 1:length(key_vars)
    y_train = x(1:T_train, key_vars(k));
    ar_model = arima(1,0,0);
    est_model = estimate(ar_model, y_train, 'Display', 'off');
    y_forecast = forecast(est_model, H, 'Y0', y_train);
    ar1_errors(:,k) = (x_test(:, key_vars(k)) - y_forecast).^2;
end
MSFE_ar1 = mean(ar1_errors, 1);
MSFE_ratio_ar1 = MSFE_all ./ MSFE_ar1;

% --- Diebold-Mariano Tests ---
% Purpose: Test whether DF forecasts are significantly more accurate than
% benchmarks.
% Explanation: The Diebold-Mariano (DM) test compares squared forecast 
% errors: DM = mean(d_t) / (std(d_t) / sqrt(n)), where d_t is the 
% difference in squared errors. A two-sided p-value is computed using a 
% t-distribution.
n = H;  % number of out‐of‐sample forecasts

for k = 1:length(key_vars)
    % 1) DF vs Naive RW
    d_rw = squared_errors(:,k) - squared_errors_rw(:,k);
    dbar = mean(d_rw);
    sd = std(d_rw,1);                                                      % use 1/N normalization for variance
    DM_rw = dbar / (sd/sqrt(n));
    p_rw = 2*(1 - tcdf(abs(DM_rw), n-1));
    
    % 2) DF vs AR(1)
    d_ar1 = squared_errors(:,k) - ar1_errors(:,k);
    dbar1 = mean(d_ar1);
    sd1 = std(d_ar1,1);
    DM_ar1 = dbar1 / (sd1/sqrt(n));
    p_ar1 = 2*(1 - tcdf(abs(DM_ar1), n-1));
    
    fprintf('DM test (%s vs RW):  DM=%.3f, p=%.3f\n', var_names{k}, ...
        DM_rw, p_rw);
    fprintf('DM test (%s vs AR1): DM=%.3f, p=%.3f\n\n', var_names{k}, ...
        DM_ar1, p_ar1);
end

% --- Forecast-Encompassing Tests ---
% Purpose: Test whether DFTL forecasts encompass RW or AR(1) forecasts.
% Explanation: Regresses the forecast error (y - f_DF) on the difference
% (f_benchmark - f_DF). A non-significant coefficient suggests DF 
% encompasses the benchmark, containing all relevant information.
nObs = H;                      % number of forecasts
dfree = nObs - 2;              % df for t‐tests

for k = 1:length(key_vars)
    y   = x_test(:, key_vars(k));       % actuals
    f0  = EM_Forecast.yhat(:,   key_vars(k));       % DFTL forecast
    f_rw= yhat_rw(:,key_vars(k));       % RW forecast
    f_a1= y_forecast;      % AR(1) forecast

    % --- Test 1: Does DF encompass RW? regress (y - f0) on (f_rw - f0)
    d_rw = y - f0;
    X_rw = [ones(nObs,1), (f_rw - f0)];
    beta_rw = (X_rw'*X_rw)\(X_rw'*d_rw);
    res_rw  = d_rw - X_rw*beta_rw;
    sigma2_rw = (res_rw'*res_rw)/dfree;
    Vbeta_rw  = sigma2_rw * inv(X_rw'*X_rw);
    t_rw      = beta_rw(2)/sqrt(Vbeta_rw(2,2));
    p_rw      = 2*(1 - tcdf(abs(t_rw), dfree));
    
    % --- Test 2: Does DF encompass AR(1)? regress (y - f0) on (f_a1 - f0)
    d_a1 = y - f0;
    X_a1 = [ones(nObs,1), (f_a1 - f0)];
    beta_a1 = (X_a1'*X_a1)\(X_a1'*d_a1);
    res_a1  = d_a1 - X_a1*beta_a1;
    sigma2_a1 = (res_a1'*res_a1)/dfree;
    Vbeta_a1  = sigma2_a1 * inv(X_a1'*X_a1);
    t_a1      = beta_a1(2)/sqrt(Vbeta_a1(2,2));
    p_a1      = 2*(1 - tcdf(abs(t_a1), dfree));

    fprintf('\nEncompassing tests for %s:\n', var_names{k});
    fprintf(' DFTL vs RW:  \tβ̂=%.3f, t=%.2f, p=%.3f\n', beta_rw(2), t_rw, ...
        p_rw);
    fprintf(' DFTL vs AR1: \tβ̂=%.3f, t=%.2f, p=%.3f\n', beta_a1(2), t_a1, ...
        p_a1);
end

% --- Ljung-Box Test ---
% Purpose: Test for autocorrelation in forecast residuals.
% Explanation: The Ljung-Box test computes a Q-statistic for residual 
% autocorrelations up to maxLags, testing the null hypothesis of no 
% autocorrelation against the alternative of serial correlation.
if H > 1
maxLags = ceil(H/3);                                                       % how many lags to test
alpha   = 0.05;                                                            % significance level

% Compute residuals matrix (H×4)
residuals = x_test(:, key_vars) - EM_Forecast.yhat(:, key_vars);

for k = 1:length(key_vars)
    res_k = residuals(:,k);
    n     = length(res_k);

    % compute autocorrelations up to maxLags
    acfAll = autocorr(res_k, 'NumLags', maxLags);
    % autocorr returns [lag0, lag1, …, lagMaxLags]
    rho = acfAll(2:end);   % drop lag-0

    % Ljung–Box Q statistic
    Q = n*(n+2) * sum( rho.^2 ./ (n - (1:maxLags))' );
    pValue = 1 - chi2cdf(Q, maxLags);

    % decision
    if pValue < alpha
        verdict = 'Reject H_0 → autocorrelation';
    else
        verdict = 'Fail to reject H_0';
    end

    fprintf('%s: Q(%d)=%.2f, p=%.3f → %s\n', ...
        var_names{k}, maxLags, Q, pValue, verdict);
end
else
end
% Helper for printing
function s = ternary(cond, a, b)
    if cond, s = a; else s = b; end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Visualize forecasting performance and model diagnostics.
% Explanation: Creates a multi-panel figure showing MSFE, actual vs. 
% forecasted values, factor time series, cross-correlations, and MSFE 
% ratios, aiding interpretation of DF performance and factor dynamics.

% --- Cross-Correlation of Factors ---
% Purpose: Identify the most correlated pair of factors.
% Explanation: Computes the correlation matrix of estimated factors and 
% finds the pair with the highest absolute correlation, useful for 
% understanding factor relationships.
corr_matrix = corr(EM.F);
lower_tri = tril(corr_matrix, -1);
[~, max_idx] = max(abs(lower_tri(:)));
[i, j] = ind2sub(size(lower_tri), max_idx);
disp(['Most correlated factors: ', num2str(i), ' & ', num2str(j)]);
max_lags = 5;
[cross_corr, lags] = xcorr(EM.F(:,i), EM.F(:,j), max_lags, 'coeff');
figure('Position', [100 100 1200 800]);

% Plot MSFE over horizons
subplot(3,4,1);
plot(1:H, MSFE_horizon, '-o', 'LineWidth', 1.5);
title('MSFE per Horizon'); xlabel('Horizon'); ylabel('MSFE'); grid on;

% Actual vs Forecast with CI for each key variable
for idx = 1:4
    subplot(3,4,1+idx);
    plot(1:T_train, x(1:T_train, key_vars(idx)), 'b-', 'DisplayName', ...
        'Train Actual'); hold on;
    plot(T_train+1:T_train+H, x_test(:, key_vars(idx)), 'k-', 'DisplayName', ...
        'Test Actual');
    plot(T_train+1:T_train+H, EM_Forecast.yhat(:, key_vars(idx)), 'r--', ...
        'DisplayName', 'Forecast');
    hold off; title(var_names{idx});
    xlabel('Time'); ylabel('Value'); grid on;
    if idx==1, legend('Location', 'Best'); end
end

% Histogram of MSFE across variables
subplot(3,4,6);
histogram(MSFE_all, 10);
title('MSFE Distribution'); xlabel('MSFE'); ylabel('Frequency'); grid on;

% Time series of latent factors
subplot(3,4,7);
plot(EM.F, 'LineWidth', 1.5);
title('Latent Factors over Time'); xlabel('Time'); ylabel('Factor Value');
legend(arrayfun(@(x) sprintf('F%d', x), 1:r, 'UniformOutput', false)); 
grid on;

% Cross-correlogram of most correlated pair
subplot(3,4,8);
stem(lags, cross_corr, 'LineWidth', 1.5);
title(['Cross-Correlogram: Factor ' num2str(i) ' vs ' num2str(j)]);
xlabel('Lag'); ylabel('Corr'); grid on;

sgtitle(sprintf('QML_DFM Forecasting: p=%d', p));

% 6. MSFE Ratio Bar Chart
subplot(3,4,9);
bar([MSFE_ratio_rw; MSFE_ratio_ar1]');
title('MSFE Ratios: DFTL vs RW and AR(1)');
xticklabels(var_names); ylabel('Ratio');
legend('DFTL / RW', 'DFTL / AR(1)', 'Location', 'Best'); grid on;

sgtitle(sprintf('QML_DFM Forecasting Performance: p=%d', p));

% MSFE per Horizon for Key Variables (Individual)
subplot(3,4,10);
plot(1:H, squared_errors(:,1), '-o', 'LineWidth', 1.3); hold on;
plot(1:H, squared_errors(:,2), '-s', 'LineWidth', 1.3);
plot(1:H, squared_errors(:,3), '-d', 'LineWidth', 1.3);
plot(1:H, squared_errors(:,4), '-^', 'LineWidth', 1.3);
hold off;
title('MSFE per Horizon for Key Variables');
xlabel('Horizon'); ylabel('Squared Error');
legend(var_names, 'Location', 'Best'); grid on;

sgtitle(sprintf('QML_DFM Forecasting: p=%d', p));

disp('Forecasting and visualization complete.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Forecast Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [EM_Forecast] = DFM_forecast(EM, h, mean_train, std_train)
    % Out-of-sample DFM forecast using EM outputs
    N = size(EM.Lambda,1);
    pr = size(EM.A,1);
    A = EM.A; Lambda = EM.Lambda; Q = EM.Q; R = EM.R;
    currState = EM.xitT(end,1:pr)'; currP = EM.PtT(:,:,end);
    forecast = zeros(h, N);
    states   = zeros(h, pr);
    feVar    = zeros(N, N, h);
    for t = 1:h
        nextState = A * currState;
        Pnext     = A * currP * A' + Q;
        forecast(t,:)    = (Lambda * nextState)';
        feVar(:,:,t)     = Lambda * Pnext * Lambda' + R;
        states(t,:)      = nextState';
        currState = nextState;
        currP     = Pnext;
    end
    EM_Forecast.yhat_norm=forecast;
    EM_Forecast.states_forecast=states;
    EM_Forecast.forecast_error_var=feVar;
    EM_Forecast.yhat=forecast .* std_train + mean_train;
end
