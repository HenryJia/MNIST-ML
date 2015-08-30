function [convTheta1, convBias1, convTheta2, convBias2, layer1, layer2, layer3, layer4] = leNet5Forward(x)

%x = rowToMat(...);

for i = 1:6
  convTheta1(:, :, 1, i) = randInitializeWeights(1, 6, 5, 5);
end
for i = 1:6
  for j = 1:16
    convTheta2(:, :, i, j) = randInitializeWeights(6, 16, 5, 5);
  end
end

convBias1 = randInitializeWeights(1, 6, 6, 1);
convBias2 = randInitializeWeights(6, 16, 16, 1);

layer1 = cnnConvolve(5, 6, x, convTheta1, convBias1);
for i = 1:6
  layer2(:, :, i) = maxPool(layer1(:, :, i), 2, 2);
end

layer3 = cnnConvolve(5, 16, layer2, convTheta2, convBias2);
for i = 1:16
  layer4(:, :, i) = maxPool(layer3(:, :, i), 2, 2);
end

end