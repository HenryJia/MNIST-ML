function [img, layer1, layer2] = test(x)

for i = 1:6
  theta{i} = randInitializeWeights(4, 5);
end

bias = randInitializeWeights(5, 1);

img{1} = rowToMat(x);

layer1 = convForward(img, theta, bias, 1, 6);

%layer2 = maxPool(layer1, 2, 2);

figure(1)
imshow(img{1}, [min(min(img{1})), max(max(img{1}))])
for i = 1:6
  figure(i + 1)
  imshow(layer1{i}, [min(min(layer1{i})), max(max(layer1{i}))])
end
%figure(3)
%imshow(layer2, [min(min(layer2{1})), max(max(layer2{1}))])

end