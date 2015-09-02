function [convTheta1, convBias1, convTheta2, convBias2, Theta1, Theta2, layer1, layer2, layer3, layer4, layer4Row, z2, a2, z3, a3, z4, a4, J] = leNet5(xRaw, yRaw, iters, alpha, convAlpha)

m = size(xRaw, 1);

fieldDim = 5;
l1Dim = 6;
l3Dim = 16;
l5In = 256;
l5Dim = 128;
l6Dim = 64;
outputDim = 10;

for i = 1:l1Dim
  convTheta1(:, :, 1, i) = randInitializeWeights(1, l1Dim, fieldDim, fieldDim);
end
for i = 1:l1Dim
  for j = 1:l3Dim
    convTheta2(:, :, i, j) = randInitializeWeights(l1Dim, l3Dim, fieldDim, fieldDim);
  end
end

convBias1 = randInitializeWeights(1, l1Dim, l1Dim, 1);
convBias2 = randInitializeWeights(l1Dim, l3Dim, l3Dim, 1);

Theta1 = randInitializeWeights(l5In + 1, l5Dim, l5In + 1, l5Dim);
Theta2 = randInitializeWeights(l5Dim + 1, l6Dim, l5Dim + 1, l6Dim);
Theta3 = randInitializeWeights(l6Dim + 1, outputDim, l6Dim + 1, outputDim);

for i = 1:iters
  for b = 1:1

  % Forward Prograte convolutional layers and max pooling layers
  x = rowToMat(xRaw(b, :), 28, 28);
  y = yRaw(b, :);
  layer1 = cnnConvolve(5, l1Dim, x, convTheta1, convBias1);
  layer1 = max(0, layer1);
  for j = 1:l1Dim
    [layer2(:, :, j), layer2Pos(:, :, :, j)] = maxPool(layer1(:, :, j), 2, 2);
  end

  layer3 = cnnConvolve(5, l3Dim, layer2, convTheta2, convBias2);
  layer3 = max(0, layer3);
  for j = 1:l3Dim
    [layer4(:, :, j), layer4Pos(:, :, :, j)] = maxPool(layer3(:, :, j), 2, 2);
  end

  % Forward propagate through the MLP
  layer4Row = [1, mat3DToRow(layer4)'];
  z2 = layer4Row * Theta1;
  a2 = tanh(z2);

  a2 = [1, a2];
  z3 = a2 * Theta2;
  a3 = tanh(z3);

  a3 = [1, a3];
  z4 = a3 * Theta3;
  a4 = softmax(z4);

  % Calculate cost
  J((i - 1) * iters + b) = -y * log(a4)'

  % Back propagate through the MLP
  delta4 = a4 - y;
  delta3 = (delta4 * Theta3')(:, 2:end)  .* (sech(z3) .^ 2);
  delta2 = (delta3 * Theta2')(:, 2:end)  .* (sech(z2) .^ 2);

  Delta3 = delta4' * a3;
  Delta2 = delta3' * a2;
  Delta1 = delta2' * layer4Row;

  Theta3_grad = Delta3' / m;
  Theta2_grad = Delta2' / m;
  Theta1_grad = Delta1' / m;

  % Back propagate the errors with respect to the inputs of the MLP (the output of the convolutional layers
  convdelta4Row = (delta2 * Theta1')(:, 2:end);
  for j = 1:l3Dim
    convdelta4(:, :, j) = rowToMat(convdelta4Row(1, ((j - 1) * 4 * 4 + 1):(j * 4 * 4)), 4, 4);
  end

  %convdelta4(:, :, 1:5)

  % Back propagate through the max pooling layers by setting all gradients to 0 except the units selected by max pooling.
  convdelta3 = zeros(size(layer3));
  for j = 1:l3Dim
    for k = 1:4
      for l = 1:4
        convdelta3(((k - 1) * 2 + layer4Pos(1, k, l, j)), ((l - 1) * 2 + layer4Pos(2, k, l, j)), j) = convdelta4(k, l, j);
      end
    end
  end
  convdelta3 = convdelta3 .* (layer3 > 0);
%{
  convdelta2 = zeros(size(layer2));
  for j = 1:l1Dim
    for k = 1:l3Dim
      filter = squeeze(convTheta2(:, :, j, k));
      % Flip the feature matrix because of the definition of convolution, as explained later
      %filter = rot90(squeeze(filter), 2);
      %filter = squeeze(filter);
      convdelta2(:, :, j) += conv2(squeeze(convdelta3(:, :, k)), filter, 'full');
    end
  end
%}
  convdelta2 = zeros(size(layer2));
  for j = 1:l1Dim
    for k = 1:l3Dim
      convdelta2(:, :, j) += conv2(squeeze(convdelta3(:, :, k)), squeeze(convTheta2(:, :, j, k)), 'full');
    end
  end
  convdelta1 = zeros(size(layer1));
  for j = 1:l1Dim
    for k = 1:12
      for l = 1:12
        convdelta1(((k - 1) * 2 + layer2Pos(1, k, l, j)), ((l - 1) * 2 + layer2Pos(2, k, l, j)), j) = convdelta2(k, l, j);
      end
    end
  end
  convdelta1 = convdelta1 .* (layer1 > 0);

  %convdelta3(:, :, 1)
  %layer3(:, :, 1)

  % Back propagate the gradients with respect to convTheta. This can be done by convoluting the layer before the weights with the errors of the layers after the weights
  for j = 1:l1Dim
    for k = 1:l3Dim 
      filter = convdelta3(:, :, k);
      % Flip the feature matrix because of the definition of convolution, as explained later
      filter = rot90(squeeze(filter), 2);
      %filter = squeeze(filter);
      convDelta3(:, :, j, k) = conv2(squeeze(layer2(:, :, j)), filter, 'valid');
    end
  end
  for k = 1:l1Dim 
    filter = convdelta1(:, :, k);
    % Flip the feature matrix because of the definition of convolution, as explained later
    filter = rot90(squeeze(filter), 2);
    %filter = squeeze(filter);
    convDelta1(:, :, 1, k) = conv2(squeeze(x), filter, 'valid');
  end

  % We don't need to backpropagate any more to get the gradients with respect to the biases as they're just convdelta3 summed
  for j = 1:l3Dim
    convBiasDelta2(j) = sum(sum(convdelta3(:, :, j)));
  end
  for j = 1:l1Dim
    convBiasDelta1(j) = sum(sum(convdelta1(:, :, j)));
  end

  %convBias2 = convBias2 - convAlpha * convBiasDelta;

  % Apply gradients to stochastic gradient descent
  %convTheta2 = convTheta2 - convAlpha * convDelta3;

  %figure((i - 1) * iters + b)
  %imshow(convTheta2(:, :, 1), [min(min(convTheta2(:, :, 1))), max(max(convTheta2(:, :, 1)))])

%{
  % For Gradient checking
  epsilon = 10^-5;
  %convTheta2(1, 2, 4, 15) += epsilon;
  %convTheta1(1, 2, 1, 4) += epsilon;
  %layer4(1, 2, 15) += epsilon;
  convBias1(2) += epsilon;

  layer1 = cnnConvolve(5, l1Dim, x, convTheta1, convBias1);
  layer1 = max(0, layer1);
  for j = 1:l1Dim
    [layer2(:, :, j), layer2Pos(:, :, :, j)] = maxPool(layer1(:, :, j), 2, 2);
  end

  layer3 = cnnConvolve(5, l3Dim, layer2, convTheta2, convBias2);
  layer3 = max(0, layer3);
  for j = 1:l3Dim
    [layer4(:, :, j), layer4Pos(:, :, :, j)] = maxPool(layer3(:, :, j), 2, 2);
  end

  % Forward propagate through the MLP
  layer4Row = [1, mat3DToRow(layer4)'];
  z2 = layer4Row * Theta1;
  a2 = tanh(z2);

  a2 = [1, a2];
  z3 = a2 * Theta2;
  a3 = tanh(z3);

  a3 = [1, a3];
  z4 = a3 * Theta3;
  a4 = softmax(z4);

  % Calculate cost
  test = -y * log(a4)'

  (test - J((i - 1) * iters + b)) / epsilon
  %convDelta3(:, :, 4, 15)
  %convdelta2(:, :, 1)
  %convdelta4Row(1, 17:32)
  %convdelta4(:, :, 2)
  %convDelta1(:, :, 1, 4)
  convBiasDelta1
  %size(squeeze(convdelta3))
  %size(squeeze(convdelta2))
%}
  Theta3 = Theta3 - alpha * Delta3';
  Theta2 = Theta2 - alpha * Delta2';
  Theta1 = Theta1 - alpha * Delta1';
  end
end

end