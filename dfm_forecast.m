%% dfm_forecast_accuracy.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics, University of Bologna
% Description:
%   Evaluates out-of-sample forecast accuracy for the QML Dynamic Factor Model
%   estimates generated via BL_Estimate. Computes mean squared forecast errors,
%   constructs 95% confidence intervals using forecast error variances, and
%   visualizes forecast performance alongside latent factor dynamics.

%% Initialization
clear; close all; clc;

%% Dataset Selection
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg(...
    'PromptString', 'Select the dataset to load:',...
    'SelectionMode', 'single',...
    'ListString', options,...
    'Name', 'Dataset Selection',...
    'ListSize', [400 200]);
if ~ok
    error('No dataset selected. Exiting...');
end

%% Load Data and Define Key Variables
switch choiceIndex
    case 1  % Monthly data
        filepath = '/Users/moka/Research/Thesis/Live Project/Processed_Data/MD1959.xlsx';
        raw = readtable(filepath);
        x = table2array(raw);
        T = size(x,1);
        key_vars = [1, 24, 105, 77];  % GDP, Unemp, Inflation, Interest Rate
    case 2  % Quarterly data
        filepath = '/Users/moka/Research/Thesis/Live Project/Processed_Data/QD1959.xlsx';
        raw = readtable(filepath);
        x = table2array(raw(:,2:end));
        T = size(x,1);
        key_vars = [1, 58, 116, 136];
    otherwise
        error('Unexpected selection.');
end
var_names = {'GDP', 'Unemployment', 'Inflation', 'Interest Rate'};

%% Define Training Period and Horizon
defaultTrain = '670';
prompt = sprintf('Dataset has %d observations. Enter training size (T_train):', T);
userIn = inputdlg(prompt, 'Input T_train', [3 100], {defaultTrain});
if isempty(userIn)
    error('No input given. Exiting...');
end
T_train = str2double(userIn{1});
assert(T_train>0 && T_train<T, 'T_train must be between 1 and %d', T-1);

h = input('Enter forecast horizon h (1 to 24): ');
assert(h>=1 && h<=24 && floor(h)==h, 'h must be an integer between 1 and 24');

%% Prepare Data Subsets
x_train = x(1:T_train, :);
x_test  = x(T_train+1:T_train+h, :);
mean_train = mean(x_train);
std_train  = std(x_train);
x_train_norm = (x_train - mean_train) ./ std_train;
x_test_norm  = (x_test  - mean_train) ./ std_train;

%% Load DFM Forecast Outputs
% EM structure must be in workspace from BL_Estimate
[forecast, states_fcast, forecast_error_var] = DFM_out_of_sample_forecast(EM, h);

%% Forecast Accuracy Metrics
% Mean Squared Forecast Error for key variables
sq_err = (x_test_norm(:, key_vars) - forecast(:, key_vars)).^2;
MSFE_horizon = mean(sq_err, 2);
MSFE_var     = mean(sq_err, 1);
MSFE_overall = mean(MSFE_var);

fprintf('Overall MSFE: %.6f\n', MSFE_overall);
for k = 1:length(key_vars)
    fprintf('MSFE %s: %.6f\n', var_names{k}, MSFE_var(k));
end

% 95% Confidence Intervals using forecast_error_var
ci_lower = zeros(h, length(key_vars));
ci_upper = zeros(h, length(key_vars));
for k = 1:length(key_vars)
    idx = key_vars(k);
    vars_t = squeeze(forecast_error_var(idx, idx, 1:h));
    ci_lower(:, k) = forecast(:, idx) - 1.96 * sqrt(vars_t);
    ci_upper(:, k) = forecast(:, idx) + 1.96 * sqrt(vars_t);
end

% Root MSFE in original scale
RMSE = sqrt(MSFE_var) .* std_train(key_vars);
for k = 1:length(key_vars)
    fprintf('RMSE %s: %.4f\n', var_names{k}, RMSE(k));
end

%% Visual Diagnostics
r = size(EM.F, 2);
corrF = corr(EM.F);
[~, linearIdx] = max(abs(tril(corrF, -1)(:)));
[i, j] = ind2sub([r r], linearIdx);

figure('Position', [100 100 1200 800]);

% Plot MSFE over horizons
subplot(3,3,1);
plot(1:h, MSFE_horizon, '-o', 'LineWidth', 1.5);
title('MSFE per Horizon'); xlabel('Horizon'); ylabel('MSFE'); grid on;

% Actual vs Forecast with CI for each key variable
for k = 1:length(key_vars)
    subplot(3,3,1+k);
    tVec = T_train+1:T_train+h;
    plot(1:T_train, x_train(:, key_vars(k)), 'b-'); hold on;
    plot(tVec, x_test_norm(:, key_vars(k)), 'k-');
    plot(tVec, forecast(:, key_vars(k)), 'r--');
    fill([tVec fliplr(tVec)], [ci_lower(:,k)' fliplr(ci_upper(:,k)')], 'r', ...
         'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold off;
    title(var_names{k}); xlabel('Time'); ylabel('Value'); grid on;
end

% Histogram of MSFE across variables
subplot(3,3,6);
histogram(MSFE_var, 10);
title('MSFE Distribution'); xlabel('MSFE'); ylabel('Frequency'); grid on;

% Factor correlation heatmap\subplot(3,3,7);
imagesc(corrF); colorbar;
title('Factor Correlation Heatmap'); xlabel('Factor'); ylabel('Factor');

% Time series of latent factors\subplot(3,3,8);
plot(EM.F, 'LineWidth', 1.5);
title('Latent Factors over Time'); xlabel('Time'); ylabel('Factor Value');
legend(arrayfun(@(x) sprintf('F%d', x), 1:r, 'UniformOutput', false)); grid on;

% Cross-correlogram for most correlated pair
subplot(3,3,9);
stem(xcorr(EM.F(:,i), EM.F(:,j), 10, 'coeff'), 'LineWidth', 1.5);
title(sprintf('XCorr F%d vs F%d', i, j)); xlabel('Lag'); ylabel('Correlation'); grid on;

sgtitle(sprintf('DFM Forecast Accuracy (T_train=%d, h=%d)', T_train, h));

%% DFM_out_of_sample_forecast Helper
function [forecast, states, feVar] = DFM_out_of_sample_forecast(EM, h)
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
end
