function PredictY = forwardPropagate(X, Theta1, Theta2, Theta3)

z2 = X * Theta1;
a2 = sigmoid(z2);

a2 = [ones(length(a2), 1), a2];
z3 = a2 * Theta2;
a3 = sigmoid(z3);

a3 = [ones(length(a3), 1), a3];
z4 = a3 * Theta3;
a4 = sigmoid(z4);

[dummy, PredictY] = max(a4, [], 2);

end