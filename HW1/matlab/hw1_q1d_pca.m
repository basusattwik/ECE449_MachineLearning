close all
clearvars
clc

% Covariance matrix
S = [[12; 0; 0; 0], [0; 6; 0; 0], [0; 0; 20; 0], [0; 0; 0; 10]];

% Compute eigendecompositon
[V,d] = eig(S, 'vector');

% Cost
w = V(:, end);
J = w.' * S * w; % max value is same as max eigenvalue






