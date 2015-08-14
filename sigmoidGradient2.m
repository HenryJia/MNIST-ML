function g = sigmoidGradient2(z)

    a = sigmoid(z);
    g = a .* (1 - a) .* (1 - 2 .* a);

end