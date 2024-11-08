close all; clearvars; clc

% This is a basic implementation of the K-means algorithm
% ECE 449 Machine Learning, HW1, Q3

%% Setup

D   = 2;    % dimensions of data
K   = 3;    % number of clusters
N   = 1000;  % number of points in each cluster
maxItr = 10; % max number of iterations to run

% Initialize data blobs
std    = 0.5;  % standard deviation
mean   = [[2;2], [-2;2], [-2;-2]]; % mean of blobs
[X, C] = initBlobs(N, K, D, std, mean);
Cinit  = C; 

% Visualize
scatterColors  = {[0.3010 0.7450 0.9330], [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]};
centroidColors = {[0 0.4470 0.7410], [0.6350 0.0780 0.1840], [0 0 0]};

%% K Means

T = K * N; % total number of data points
r = zeros(K, T); % rows: clusters, cols: data assignment
d = zeros(K, T); 
Xtemp = zeros(size(X));
i = 1;

figure(1)
while i <= maxItr

    if i == 1
        hold on;
        scatter(X(1, :), X(2, :), '.', 'LineWidth', 0.5, 'MarkerEdgeColor', scatterColors{1}, 'MarkerFaceColor', scatterColors{1}); hold on;
        for k = 1:K
            plot(C(1, k), C(2, k), 'x', 'MarkerSize', 15, 'LineWidth', 2.0, 'Color', centroidColors{k}, 'DisplayName', ['Centroid ', num2str(k)]); 
        end
        hold off;
        grid on; grid minor;
        title('Clustered Data');
        legend('show');
        xlabel('x1'); ylabel('x2');
        pause(0.5)
    end

    % get distance of points from cluster centers
    for n = 1:T
        for k = 1:K
            % Calculate distance between each point and the centroids
            d(k,n) = sum((X(:,n) - C(:,k)).^2);
        end
    end

    % Assign points to clusters
    for n = 1:T
        [~, ind] = min(d(:,n));
        for k = 1:K           
             if k == ind
                 r(k,n) = 1;
             end
        end
    end

    % update centroids
    for k = 1:K
        C(:,k) = sum(r(k,:) .* X, 2) ./ sum(r(k,:));
    end

    % Show intermediate steps
    clf;
    if mod(i, 1) == 0
        hold on;
        for k = 1:K
            ctemp = find(r(k, :) == 1);
            scatter(X(1, ctemp), X(2, ctemp), '.', 'LineWidth', 0.5, 'MarkerEdgeColor', scatterColors{k}, 'MarkerFaceColor', scatterColors{k},  'DisplayName', ['Cluster ', num2str(k)]); 
            plot(C(1, k), C(2, k), 'x', 'MarkerSize', 15, 'LineWidth', 2.0, 'Color', centroidColors{k}, 'DisplayName', ['Centroid ', num2str(k)]); 
        end
        hold off;
        grid on; grid minor;
        title('Clustered Data');
        legend('show');
        xlabel('x1'); ylabel('x2');
    end
    pause(1)

    % reset assignment
    r(:) = 0;
    d(:) = 0;

    % increment i
    i = i + 1;
end

disp('Done!');