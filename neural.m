function [Theta1, Theta2, Theta3] = neural(X, YRaw)

% Load and normalise data

fprintf('Load Data\n');

%X = load("trainXnonZero.csv");
%YRaw = load("trainY.csv");;

m = length(YRaw);
I = eye(10);
Y = zeros(m, 10);
for i=1:m
    Y(i, :) = I((YRaw(i, :) + 1), :);
end
% Set Important Variables:

alpha = 0.05;
mu = 0.1;
lambda = 0;
iters = 1;
scatterIters = 2;

fprintf('Data Loaded. Normalised Features And Add Bias Units. Press Enter\n');

X_norm =  featureNormalize(X);
%X_norm = X;

%Add The Bias Units

fprintf('Features Normalised. Add Bias Units. Press Enter\n');

X_norm = [ones(size(X_norm, 1) ,1), X_norm];

fprintf('Features Normalised. Initialise Thetas. Press Enter\n');

%Theta1 = abs(randInitializeWeights(784, 500));
%Theta2 = abs(randInitializeWeights(500, 300));
%Theta3 = abs(randInitializeWeights(300, 10));
Theta1 = randInitializeWeights(784, 500);
Theta2 = randInitializeWeights(500, 300);
Theta3 = randInitializeWeights(300, 10);
Theta1(1:5, 1:5)
Theta2(1:5, 1:5)
Theta3(1:5, 1:5)

% Calculate Thetas & Results For First Hidden Layer

fprintf('Thetas initialise. Training. Press Enter\n');

tic
[Theta1, Theta2, Theta3] = train(X_norm, Y, Theta1, Theta2, Theta3, alpha, lambda, iters, scatterIters);
toc

fprintf('Training Complete. Calculate Costs. Press Enter\n');
%pause;

predictTrain = forwardPropagate(X_norm, Theta1, Theta2, Theta3);

%J = sum(sum((predictTrain - Y) .^ 2))/ (2 * m);

% Add on the penalty for regularization
%J += (sum(sum(Theta1(2:end, :) .^ 2)) + sum(sum(Theta2(2:end, :) .^ 2)) + sum(sum(Theta3(2:end, :) .^ 2))) * (lambda / (2 * m))

fprintf('Error\n');

error = 0;
for i=1:m
    if(predictTrain(i, :) != YRaw(i, :))
        error = error + 1;
    end
end

error = error / m

fprintf('Done. Press Enter\n');