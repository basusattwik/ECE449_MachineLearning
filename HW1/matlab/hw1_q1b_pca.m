close all
clearvars
clc

% data points
X = [[1;3], [4;7]];

% visualize
plot(X(1,:), X(2,:), '.', 'MarkerSize', 30); grid on; grid minor;
xlabel('x1'); ylabel('x2');
xlim([0, 5]); ylim([0,8]);

mu = mean(X,2);
Xb = X - mu; % mean centered
S  = 0.5 * (Xb * Xb.'); % covariance matrix

[V,D] = eig(S);




