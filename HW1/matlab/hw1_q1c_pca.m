close all
clearvars
clc

% data points
X = [[2;0], [2;2], [6;0], [6;2]];

% visualize
plot(X(1,:), X(2,:), '.', 'MarkerSize', 30); grid on; grid minor;
xlabel('x1'); ylabel('x2');
xlim([0, 7]); ylim([0,7]);

mu = mean(X,2);
Xb = X - mu; % mean centered
S  = 0.5 * (Xb * Xb.'); % covariance matrix

[V,D] = eig(S);




