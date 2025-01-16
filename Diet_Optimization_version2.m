% Sustainable Diet Optimization Framework
% -------------------------------------------------------------------------
% Author: Le Ma
% Location: [Room 601, Physics School, USYD, Sydney]
% Date: [16/01/2025]
% Version: 2.0  多目标优化
% -------------------------------------------------------------------------
% Description:
% This MATLAB script implements a multi-objective optimization framework 
% to design sustainable diets for future,likes 2050. The framework addresses three key 


% Initialization
clear; clc; close all;

% Data Loading
disp('Loading multi-year data...');
years = 1980:2020; % Time range
num_years = length(years);

% Constants
num_countries = 164; % Number of countries
num_nutrients = 22; % Number of nutrient types
n = 7793; % Number of food types

% Initialize multi-year matrices
Y = cell(num_years, 1); % Final demand matrix for each year
x = cell(num_years, 1); % Total output for each year
m = cell(num_years, 1); % Carbon intensity for each year
q = cell(num_years, 1); % Carbon coefficients for each year
L = cell(num_years, 1); % Leontief matrix for each year
D = zeros(num_years, num_countries, num_nutrients); % Annual demand
S = zeros(num_years, num_countries, num_nutrients); % Annual production

% Load data for each year
for t = 1:num_years
    year = years(t);
    Y{t} = load(['Y_', num2str(year), '.mat']); % Final demand matrix
    x{t} = load(['x_', num2str(year), '.mat']); % Total output
    m{t} = load(['m_', num2str(year), '.mat']); % Carbon intensity
    q{t} = load(['q_', num2str(year), '.mat']); % Carbon coefficients
    L{t} = load(['L_', num2str(year), '.mat']); % Leontief matrix
    D(t, :, :) = reshape(Y{t}.v1, [num_countries, num_nutrients]); % Demand
    S(t, :, :) = reshape(x{t}.v1, [num_countries, num_nutrients]); % Supply
end

disp('Multi-year data loading completed.');

% Load historical diet data
data = load('historical_diet_data.mat'); % Assume variable name is `historical_diet`
historical_diet = data.historical_diet;
disp('Historical diet data loaded.');

[num_countries, num_food_types, num_years] = size(historical_diet);

% Initialize change rate range
historical_change = zeros(num_countries, num_food_types, 2); % [Min, Max]

% Calculate historical change rate for each country and food type
for country = 1:num_countries
    for food = 1:num_food_types
        changes = zeros(num_years - 1, 1);
        for year = 1:(num_years - 1)
            if historical_diet(country, food, year) > 0
                changes(year) = (historical_diet(country, food, year+1) - historical_diet(country, food, year)) / ...
                                 historical_diet(country, food, year);
            else
                changes(year) = 0; % Define change rate as 0 if the historical consumption is 0
            end
        end
        historical_change(country, food, 1) = min(changes); % Minimum change rate
        historical_change(country, food, 2) = max(changes); % Maximum change rate
    end
end

% Initialize cultural acceptability range
acceptability_range = zeros(num_countries, num_food_types, 2); % [Min, Max]

% Calculate cultural acceptability based on historical diet
for country = 1:num_countries
    for food = 1:num_food_types
        food_share = squeeze(historical_diet(country, food, :)) ./ ...
                     sum(historical_diet(country, :, :), 2); % Proportion of food type
        mean_share = mean(food_share); % Mean share
        std_share = std(food_share); % Standard deviation of share
        acceptability_range(country, food, 1) = max(0, mean_share - 1.96 * std_share); % Lower bound
        acceptability_range(country, food, 2) = mean_share + 1.96 * std_share; % Upper bound
    end
end

% Save the calculated data as CSV files
csvwrite('historical_change.csv', historical_change);
csvwrite('acceptability_range.csv', acceptability_range);

% Load additional data
Nutrients_per_thousand_USD = csvread('Nutrients_per_thousand_USD.csv');
economic_data = csvread('economic_data.csv'); % [num_years, num_countries, GDP_per_capita, PPP_index]
GDP_per_capita = economic_data(:, :, 2);
PPP_index = economic_data(:, :, 3);
food_prices = csvread('food_prices.csv');
environmental_footprint_data = load('environmental_footprint_data.mat'); % [num_years, num_countries, num_food_types, 6]
historical_change = csvread('historical_change.csv'); % Historical change rate range
acceptability_range = csvread('acceptability_range.csv'); % Cultural acceptability range

% Food industry index (assuming the first 10 rows correspond to the food industry)
food_indices = [1:10];
population = csvread('population_data.csv'); % Population data [num_years × num_countries]

% Calculate observed current dietary intake (ensure consistency with historical_diet dimensions)
Q_obs = zeros(num_countries, num_food_types); % Matching historical_diet dimensions
for t = 1:num_years
    F_household = squeeze(Y{t}.household_consumption(food_indices, :)); % Extract household consumption
    for country = 1:num_countries
        % Per capita food intake (adjusted for food types)
        Q_obs(country, :) = F_household(:, country) ./ population(t, country); 
    end
end

% Calculate economic weights: Inverse of GDP per capita and PPP index
economic_weights = 1 ./ (GDP_per_capita .* PPP_index);

% Unit weight of food prices
price_weights = food_prices; % Assuming unit is Price_per_gram

% Nutritional constraints (upper and lower bounds for nutrient intake)
nutrition_constraints = [52, 2300, 3200, 65, 23, 125, 29, ...
                         520, 17, 3247, 1.1, 1.1, 364, 6.1, 2.2, ...
                         757, 0.8, 2, 55, 14, 4.7, 1.2];
% Nutritional constraints for optimization model
% Variables represent the following nutrients (in order):
% [Protein, Caloric Lower Bound, Caloric Upper Bound, Total Fat, Saturated Fat, 
%  Sugar, Fiber, Calcium, Iron, Potassium, Thiamine, Riboflavin, Folate, Zinc, 
%  Vitamin B12, Phosphorus, Copper, Manganese, Selenium, Polyunsaturated Fatty Acids, 
%  Pantothenic Acid, Vitamin B6]

% Environmental footprint constraints
environmental_limits = [1866, 786, 5.01, 27.4, 6.35, 1.5]; % [Carbon, Water, Land, Nitrogen, Phosphorus, Others]

% Acceptability limits for dietary changes
acceptability_limits = [0.1, 0.95, 0.8, 1.2];

% Load nutritional content data
disp('Loading nutritional content data...');
nutrition_content = csvread('nutrition_content.csv', 1, 1); % Skip header row and first column
disp('Nutritional content data loaded.');

% Load environmental impact data
disp('Loading environmental impact data...');
environmental_impact = csvread('environmental_impact.csv', 1, 1); % Skip header row
disp('Environmental impact data loaded.');

% 定义优化变量范围
lb = reshape(acceptability_range(:, :, 1), 1, []); % 展平成 1-D 向量
ub = reshape(acceptability_range(:, :, 2), 1, []); % 展平成 1-D 向量

% 多目标优化：定义目标函数
objective = @(x) multi_objective_function(x, num_years, num_food_types, Q_obs, ...
    nutrition_content, economic_weights, price_weights, acceptability_range);

% 非线性约束：营养和环境约束
nonlin_constraints = @(x) nonlinear_constraints_with_culture(x, num_years, ...
    Q_obs, nutrition_content, nutrition_constraints, environmental_impact, ...
    environmental_limits, acceptability_range, historical_change, num_countries, num_food_types);

% 使用 gamultiobj 参数
options = optimoptions('gamultiobj', ...
    'PopulationSize', 200, ...
    'MaxGenerations', 100, ...
    'Display', 'iter', ...
    'PlotFcn', {@gaplotpareto}); % 自动绘制帕累托前沿

% 调用 gamultiobj
disp('Starting optimization...');
[x_opt, fval] = gamultiobj(objective, numel(lb), [], [], [], [], lb, ub, nonlin_constraints, options);
disp('Optimization completed.');

% 绘制帕累托前沿
figure;
plot(fval(:, 1), fval(:, 2), 'o');
xlabel('Objective 1: Diet Deviation');
ylabel('Objective 2: Environmental Impact');
title('Pareto Front');
grid on;

function f = multi_objective_function(x, num_years, num_food_types, Q_obs, ...
    nutrition_content, economic_weights, price_weights, acceptability_range)

    % 将 x 解析为多年的饮食数据
    Q_opt = reshape(x, num_years, num_food_types);

    % 初始化目标值
    diet_deviation = 0;
    environmental_impact = 0;
    economic_cost = 0;
    cultural_acceptability = 0;

    % 逐年计算目标值
    for t = 1:num_years
        Q_t = Q_opt(t, :);

        % 目标 1：饮食偏差
        diet_deviation = diet_deviation + sum((Q_t - Q_obs).^2);

        % 目标 2：环境足迹
        environmental_impact = environmental_impact + sum(Q_t .* price_weights);

        % 目标 3：经济权重
        economic_cost = economic_cost + sum((nutrition_content * Q_t) .* economic_weights);

        % 目标 4：文化可接受性
        cultural_acceptability = cultural_acceptability + ...
            sum(max(0, Q_t - acceptability_range(:, :, 2)).^2 + ...
                max(0, acceptability_range(:, :, 1) - Q_t).^2);
    end

    % 返回 4 个目标的值
    f = [diet_deviation, environmental_impact, economic_cost, cultural_acceptability];
end

function [c, ceq] = nonlinear_constraints_with_culture(x, num_years, ...
    Q_obs, nutrition_content, nutrition_constraints, environmental_impact, ...
    environmental_limits, acceptability_range, historical_change, num_countries, num_food_types)

    % 初始化非线性约束
    c = [];
    ceq = [];

    % 将 x 转换为二维形式
    Q_opt = reshape(x, num_years, num_food_types);

    % 遍历每年，逐年检查约束
    for t = 1:num_years
        % 当前年份的饮食变量
        Q_t = Q_opt(t, :);

        %% Nutritional Constraints: 营养需求约束
        nutrients = Q_t * nutrition_content; % 每种营养的总摄入
        c_nutrition = [nutrition_constraints(1:7) - nutrients(1:7), ... % 最低摄入要求
                       nutrients(8:end) - nutrition_constraints(8:end)]; % 最高摄入限制

        %% Environmental Constraints: 环境足迹约束
        c_environment = sum(environmental_impact(t, :, :) .* Q_t, 2) - environmental_limits; % 每种环境足迹的限制

        %% Cultural Acceptability Constraints: 文化可接受性约束
        c_acceptability = [
            Q_t - reshape(acceptability_range(:, :, 2), 1, []); ... % 超出文化上界
            reshape(acceptability_range(:, :, 1), 1, []) - Q_t];   % 低于文化下界

        %% Combine all constraints
        c = [c; c_nutrition(:); c_environment(:); c_acceptability(:)];
    end

    % No equality constraints
    ceq = [];
end
