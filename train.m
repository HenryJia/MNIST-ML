function [TrainedTheta1, TrainedTheta2, TrainedTheta3] = train(X, Y, Theta1, Theta2, Theta3, alpha, lambda, iters, layer3train, max)

m = size(X, 1);
mu = 10^-10;
JAll = zeros(iters);

TrainedTheta1 = Theta1;
TrainedTheta2 = Theta2;
TrainedTheta3 = Theta3;

for i = 1:iters

    % Forward Propagation
    z2 = X * TrainedTheta1;
    %a2 = sigmoid(z2);
    a2 = tanh(z2);
    z2(1:5, 1:5)

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    %a3 = sigmoid(z3);
    a3 = tanh(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    %a4 = sigmoid(z4);
    a4 = softmax(z4);
        
    % Cost Function
    %J = sum(sum((log(a4) .* Y + log(1 - a4) .* (1 - Y))/(-m)))
    J = -sum(sum(log(a4) .* Y)) / m
    JAll(i) = J;

    % Calculate small delta
    delta4 = a4 - Y;
    %delta4 = Y - a4;
    %delta3 = (delta4 * TrainedTheta3')(:, 2:end)  .* sigmoidGradient(z3);
    delta3 = (delta4 * TrainedTheta3')(:, 2:end)  .* (sech(z3) .^ 2);

    %delta2 = (delta3 * TrainedTheta2')(:, 2:end)  .* sigmoidGradient(z2);
    delta2 = (delta3 * TrainedTheta2')(:, 2:end)  .* (sech(z2) .^ 2);

    % Accumulate small delta to calculate big delta which is the partial derivatives
    Delta3 = delta4' * a3;
    Delta2 = delta3' * a2;
    Delta1 = delta2' * X;

    % Finish off the calculation and add on the penalty term for regularization
    Theta3_grad = Delta3' / m;
    Theta2_grad = Delta2' / m;
    Theta1_grad = Delta1' / m;
%{  
    Gradient Checking for z4
    epsilon = 10^-5;
    testJ1 = J;
    z4(1,2) += epsilon;
    %a4 = sigmoid(z4);
    a4 = softmax(z4);
    
    testJ2 = -sum(sum(log(a4) .* Y)) / m
    %testJ2 = sum(sum((log(a4) .* Y + log(1 - a4) .* (1 - Y))/(-m)))

    ((testJ2 - testJ1) / epsilon)
    delta4(1,2) / m
    
    Result:

Load Data
Data Loaded. Normalised Features And Add Bias Units. Press Enter
Features Normalised. Add Bias Units. Press Enter
Features Normalised. Initialise Thetas. Press Enter
Thetas initialise. Training. Press Enter
J =  2.3229
testJ2 =  2.3229
ans =    2.5155e-06
ans =    2.5155e-06
Elapsed time is 11.2056 seconds.
Training Complete. Calculate Costs. Press Enter
Error
error =  0.88702
    
%}
%{
    epsilon = 10^-5;
    testJ1 = J;
    
    TrainedTheta2(1,2) += epsilon;
    % Forward Propagation
    z2 = X * TrainedTheta1;
    %a2 = sigmoid(z2);
    a2 = tanh(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    %a3 = sigmoid(z3);
    a3 = tanh(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    %a4 = sigmoid(z4);
    a4 = softmax(z4);
    
    testJ2 = -sum(sum(log(a4) .* Y)) / m
    %testJ2 = sum(sum((log(a4) .* Y + log(1 - a4) .* (1 - Y))/(-m)))
    
    ((testJ2 - testJ1) / epsilon)
    Delta2'(1,2) / m

Results:
Load Data
Data Loaded. Normalised Features And Add Bias Units. Press Enter
Features Normalised. Add Bias Units. Press Enter
Features Normalised. Initialise Thetas. Press Enter
Thetas initialise. Training. Press Enter
J =  2.3393
testJ2 =  2.3393
ans = -0.010105
ans = -0.010106
Elapsed time is 9.1853 seconds.
Training Complete. Calculate Costs. Press Enter
Error
error =  0.90458
%}
    TrainedTheta1 = TrainedTheta1 - alpha * Theta1_grad;
    TrainedTheta2 = TrainedTheta2 - alpha * Theta2_grad;
    TrainedTheta3 = TrainedTheta3 - alpha * Theta3_grad;
    %Theta3_grad(1, 1:5)
%{
    %hessdelta4 = ones(m, 10);
    hessdelta4 = a4 .* (1 - a4);
    %hessDelta3 = hessdelta4' * (a3 .^ 2) / m;

    % Backpropagate. This has been checked by gradient checking and is correct
    sigmoidGradient(z3)(1:5, 1:5)
    (sigmoidGradient(z3) .^ 2)(1:5, 1:5)
    (sigmoidGradient(z2) .^ 2)(1:5, 1:5)
    hessdelta3 = (sigmoidGradient(z3) .^ 2) .* (hessdelta4 * (TrainedTheta3 .^ 2)')(:, 2:end) + sigmoidGradient2(z3) .* (delta4 * TrainedTheta3')(:, 2:end);
    hessDelta2 = hessdelta3' * (a2 .^ 2) / m;

    hessdelta2 = (sigmoidGradient(z2) .^ 2) .* (hessdelta3 * (TrainedTheta2 .^ 2)')(:, 2:end) + sigmoidGradient2(z2) .* (delta3 * TrainedTheta2')(:, 2:end);
    hessDelta1 = hessdelta2' * (X .^ 2) / m;
    
    %step1 = Theta1_grad ./ (abs(hessDelta1)' + mu);
    %sum(sum(step1))
    %csvwrite("step1.csv", step1);

    %hessdelta4 = ones(m, 10);
    %hessdelta4 = a4 .* (1 - a4);
    %hessDelta3 = hessdelta4' * (a3 .^ 2) / m;

    % Backpropagate. This has been checked by gradient checking and is correct
    %hessdelta3 = (sigmoidGradient(z3) .^ 2) .* (hessdelta4 * (TrainedTheta3 .^ 2)')(:, 2:end) + sigmoidGradient2(z3) .* (delta4 * TrainedTheta3')(:, 2:end);
    %hessDelta2 = hessdelta3' * (a2 .^ 2) / m;

    %hessdelta2 = (sigmoidGradient(z2) .^ 2) .* (hessdelta3 * (TrainedTheta2 .^ 2)')(:, 2:end) + sigmoidGradient2(z2) .* (delta3 * TrainedTheta2')(:, 2:end);
    %hessDelta1 = hessdelta2' * (X .^ 2) / m;
    
    %step2 = Theta1_grad ./ (abs(hessDelta1)' + mu);
    %sum(sum(step2))
    %csvwrite("step2.csv", step2);
    
    % Newtonian Step
    TrainedTheta1 = TrainedTheta1 - (Theta1_grad ./ (abs(hessDelta1)' + mu));
    TrainedTheta2 = TrainedTheta2 - (Theta2_grad ./ (abs(hessDelta2)' + mu));
%}
end
%figure(10)
%plot(JAll)
end
