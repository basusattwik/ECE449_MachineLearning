function [X, C] = initBlobs(N, K, D, std, mu)
%INITRANDOMDATA Summary of this function goes here
%   Detailed explanation goes here

% Create two clusters using random data
% X = [std * randn(2, N) + [2; -2], std * randn(2, N) + [-2; 2], std * randn(2, N) + [0; 0]];

X = zeros(D, K*N);
j = 1:N;
for k = 1:K
    X(:,j) = std * randn(D, N) + mu(:, k);
    j = j + N;
end
% Initialize centroids
C = [[2; 2], [-2; -2]];%, [1; -1]];
% C = 2 * randn(D, K);

end