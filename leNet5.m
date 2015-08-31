function [convTheta1, convBias1, convTheta2, convBias2, Theta1, Theta2, layer1, layer2, layer3, layer4, layer4Row, z2, a2, z3, a3, J] = leNet5(xRaw, yRaw, iters)

m = size(xRaw, 1);

for i = 1:4
  convTheta1(:, :, 1, i) = randInitializeWeights(1, 4, 5, 5);
end
for i = 1:4
  for j = 1:6
    convTheta2(:, :, i, j) = randInitializeWeights(4, 6, 5, 5);
  end
end

convBias1 = randInitializeWeights(1, 4, 4, 1);
convBias2 = randInitializeWeights(4, 6, 6, 1);

Theta1 = randInitializeWeights(96 + 1, 48, 96 + 1, 48);
Theta2 = randInitializeWeights(48 + 1, 10, 48 + 1, 10);

for i = 1:iters
  %for j = 1:m
  x = rowToMat(xRaw(1, :));
  y = yRaw(1, :);
  layer1 = cnnConvolve(5, 4, x, convTheta1, convBias1);
  for i = 1:4
    layer2(:, :, i) = maxPool(layer1(:, :, i), 2, 2);
  end

  layer3 = cnnConvolve(5, 6, layer2, convTheta2, convBias2);
  for i = 1:6
    layer4(:, :, i) = maxPool(layer3(:, :, i), 2, 2);
  end

  layer4Row = [1, mat3DToRow(layer4)'];
  z2 = layer4Row * Theta1;
  a2 = tanh(z2);

  a2 = [1, a2];
  z3 = a2 * Theta2;
  a3 = softmax(z3);

  %J(i, j) = -y * log(a3)';
  J = -y * log(a3)';

  %end
end

end