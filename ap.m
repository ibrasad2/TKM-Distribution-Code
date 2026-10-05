clc;
clear;

% Load data (carbon fiber breaking stress in GPa)
data = [0.39, 0.85, 1.08, 1.25, 1.47, 1.57, 1.61, 1.61, 1.69, 1.80, ...
        1.84, 1.87, 1.89, 2.03, 2.03, 2.05, 2.12, 2.35, 2.41, 2.43, ...
        2.48, 2.50, 2.53, 2.55, 2.56, 2.56, 2.59, 2.67, 2.73, 2.74, ...
        2.79, 2.81, 2.82, 2.85, 2.87, 2.88, 2.93, 2.95, 2.96, 2.97, ...
        3.09, 3.11, 3.11, 3.15, 3.15, 3.19, 3.22, 3.22, 3.27, 3.28, ...
        3.31, 3.31, 3.33, 3.39, 3.39, 3.56, 3.60, 3.65, 3.68, 3.70, ...
        3.75, 4.20, 4.38, 4.42, 4.70, 4.90];
x = linspace(min(data), max(data), 1000);


% TKM-Weibull PDF (transmuted Kavya-Manoharan)
tkm_weibull_pdf = @(x, alpha, beta, lambda) ...
    (exp(1)/(exp(1)-1)) * wblpdf(x, alpha, beta) .* exp(-wblcdf(x, alpha, beta)) .* ...
    (1 + lambda - (2*lambda*exp(1)/(exp(1)-1)) * (1 - exp(-wblcdf(x, alpha, beta))));

% TKM-Weibull log-likelihood (negative for minimization)
tkm_weibull_ll = @(params) -sum(log(tkm_weibull_pdf(data, params(1), params(2), params(3))));

% KM-Weibull PDF (lambda = 0 case)
km_weibull_pdf = @(x, alpha, beta) ...
    (exp(1)/(exp(1)-1)) * wblpdf(x, alpha, beta) .* exp(-wblcdf(x, alpha, beta));

% KM-Weibull log-likelihood
km_weibull_ll = @(params) -sum(log(km_weibull_pdf(data, params(1), params(2))));

% Estimate parameters for Weibull distribution (baseline)
pd_Weibull = fitdist(data', 'Weibull');
alpha_W = pd_Weibull.a;  % Shape parameter
beta_W = pd_Weibull.b;   % Scale parameter

 % Initial guesses (based on Weibull MLEs)
init_alpha = pd_Weibull.a;
init_beta = pd_Weibull.b;
init_lambda = 0;  % Neutral starting point for transmutation

% Estimate TKM-Weibull parameters
params_TKM = fminsearch(tkm_weibull_ll, [init_alpha, init_beta, init_lambda]);
alpha_TKM = params_TKM(1);
beta_TKM = params_TKM(2);
lambda_TKM = params_TKM(3);

% Estimate KM-Weibull parameters (lambda = 0)
params_KM = fminsearch(@(p) km_weibull_ll([p(1), p(2)]), [init_alpha, init_beta]);
alpha_KM = params_KM(1);
beta_KM = params_KM(2);



 figure;
histogram(data, 15, 'Normalization', 'pdf', 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'k');
hold on;
plot(x, wblpdf(x, alpha_W, beta_W), 'b-', 'LineWidth', 2, 'DisplayName', 'Weibull');
plot(x, km_weibull_pdf(x, alpha_KM, beta_KM), 'g--', 'LineWidth', 2, 'DisplayName', 'KM-Weibull');
plot(x, tkm_weibull_pdf(x, alpha_TKM, beta_TKM, lambda_TKM), 'r-', 'LineWidth', 2, 'DisplayName', 'TKM-Weibull');
xlabel('Breaking Stress (GPa)');
ylabel('Probability Density');
title('Histogram with Fitted PDFs');
legend('show');
grid on;




% TKM-Weibull CDF
tkm_weibull_cdf = @(x) (1 + lambda_TKM) * (exp(1)/(exp(1)-1) * (1 - exp(-wblcdf(x, alpha_TKM, beta_TKM)))) - ...
    lambda_TKM * (exp(1)/(exp(1)-1) * (1 - exp(-wblcdf(x, alpha_TKM, beta_TKM)))).^2 ;

% KM-Weibull CDF
km_weibull_cdf = @(x) (exp(1)/(exp(1)-1) )* (1 - exp(-wblcdf(x, alpha_KM, beta_KM)));

figure;
[F_emp, x_emp] = ecdf(data);
stairs(x_emp, F_emp, 'k-', 'LineWidth', 2, 'DisplayName', 'Empirical CDF');
hold on;
plot(x, wblcdf(x, alpha_W, beta_W), 'b-', 'LineWidth', 2, 'DisplayName', 'Weibull CDF');
plot(x, km_weibull_cdf(x), 'g--', 'LineWidth', 2, 'DisplayName', 'KM-Weibull CDF');
plot(x, tkm_weibull_cdf(x), 'r-', 'LineWidth', 2, 'DisplayName', 'TKM-Weibull CDF');
xlabel('Breaking Stress (GPa)');
ylabel('Cumulative Probability');
title('Empirical vs. Theoretical CDFs');
legend('show', 'Location', 'southeast');
grid on;



% Theoretical quantiles (TKM-Weibull)
prob = (1:length(data)) / (length(data) + 1);
quant_TKM = beta_TKM * (-log(1 - ((exp(1)-1)/exp(1)) * ...
    ((1 + lambda_TKM - sqrt((1 + lambda_TKM)^2 - 4 * lambda_TKM * prob)) ./ (2 * lambda_TKM)))).^(1/alpha_TKM);

% Sort data
data_sorted = sort(data);

figure;
plot(quant_TKM, data_sorted, 'ro', 'MarkerSize', 8, 'DisplayName', 'TKM-Weibull Q-Q');
hold on;
plot([min(data), max(data)], [min(data), max(data)], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Reference Line');
xlabel('Theoretical Quantiles (TKM-Weibull)');
ylabel('Sample Quantiles');
title('Q-Q Plot for TKM-Weibull Distribution');
legend('show');
grid on;






