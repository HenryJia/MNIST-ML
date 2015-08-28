function [img, layer1, layer2] = test(x)

weights = randInitializeWeights(2, 3);

bias = randInitializeWeights(0, 1);

img = rowToMat(x);

layer1 = convForward(img, weights, bias);

layer2 = maxPool(layer1, 2, 2);

figure(1)
imshow(img, [min(min(img)), max(max(img))])
figure(2)
imshow(layer1, [min(min(layer1)), max(max(layer1))])
figure(3)
imshow(layer2, [min(min(layer2)), max(max(layer2))])

end