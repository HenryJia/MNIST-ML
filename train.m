function [TrainedTheta1, TrainedTheta2, TrainedTheta3] = train(X, Y, Theta1, Theta2, Theta3, alpha, lambda, iters, plotIters, max)

m = size(X, 1);
mu = 10^-1;
JAll = zeros(iters / plotIters, 1);

TrainedTheta1 = Theta1;
TrainedTheta2 = Theta2;
TrainedTheta3 = Theta3;

for i = 1:iters

    % Forward Propagation
    z2 = X * TrainedTheta1;
    a2 = sigmoid(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    a3 = sigmoid(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    a4 = sigmoid(z4);
        
    if(fmod(i, plotIters) == 0)
        % Cost Function
        J = sum(sum((log(a4) .* Y + log(1 - a4) .* (1 - Y))/(-m)))

        JAll(i / plotIters) = J;
    end

    % Calculate small delta
    delta4 = a4 - Y;

    delta3 = (delta4 * TrainedTheta3')(:, 2:end)  .* sigmoidGradient(z3);

    delta2 = (delta3 * TrainedTheta2')(:, 2:end)  .* sigmoidGradient(z2);

    % Accumulate small delta to calculate big delta which is the partial derivatives
    Delta3 = delta4' * a3;
    Delta2 = delta3' * a2;
    Delta1 = delta2' * X;

    % Finish off the calculation and add on the penalty term for regularization
    Theta3_grad = Delta3' / m;
    Theta2_grad = Delta2' / m;
    Theta1_grad = Delta1' / m;

    %TrainedTheta1 = TrainedTheta1 - alpha * Theta1_grad;
    %TrainedTheta2 = TrainedTheta2 - alpha * Theta2_grad;
    %TrainedTheta3 = TrainedTheta3 - alpha * Theta3_grad;

    hessdelta4 = ones(m, 10);
    %hessdelta4 = a4 .* (1 - a4);
    hessDelta3 = hessdelta4' * (a3 .^ 2) / m;

    % Backpropagate. This has been checked by gradient checking and is correct
    hessdelta3 = (sigmoidGradient(z3) .^ 2) .* (hessdelta4 * (TrainedTheta3 .^ 2)')(:, 2:end) + sigmoidGradient2(z3) .* (delta4 * TrainedTheta3')(:, 2:end);
    hessDelta2 = hessdelta3' * (a2 .^ 2) / m;

    hessdelta2 = (sigmoidGradient(z2) .^ 2) .* (hessdelta3 * (TrainedTheta2 .^ 2)')(:, 2:end) + sigmoidGradient2(z2) .* (delta3 * TrainedTheta2')(:, 2:end);
    hessDelta1 = hessdelta2' * (X .^ 2) / m;

    % Gradient checking
    epsilon = 10^-5;
    testTheta1 = Theta1_grad';
    TrainedTheta1(5,1) += epsilon;

    z2 = X * TrainedTheta1;
    a2 = sigmoid(z2);

    a2 = [ones(length(a2), 1), a2];
    z3 = a2 * TrainedTheta2;
    a3 = sigmoid(z3);
    
    a3 = [ones(length(a3), 1), a3];
    z4 = a3 * TrainedTheta3;
    a4 = sigmoid(z4);
    
    testTheta2 = (((((a4 - Y) * TrainedTheta3')(:, 2:end) .* sigmoidGradient(z3)) * TrainedTheta2')(:, 2:end) .* sigmoidGradient(z2))' * X / m;

    (testTheta2 - testTheta1)(1, 5) / epsilon

    hessDelta1'(1:10, 1:10)
    hessdelta4(1:10, 1:10)

    % Newtonian Step
    TrainedTheta1 = TrainedTheta1 - (Theta1_grad ./ (abs(hessDelta1)' + mu));
    TrainedTheta2 = TrainedTheta2 - (Theta2_grad ./ (abs(hessDelta2)' + mu));
    TrainedTheta3 = TrainedTheta3 - (Theta3_grad ./ (abs(hessDelta3)' + mu));

end
figure(10)
plot(JAll)
end
