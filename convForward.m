function outputMap = convForward(inputMap, theta, bias)

% A non-vectorised version of cnnConvolve

height = size(inputMap, 1) - (size(theta, 1) - 1);
width = size(inputMap, 2) - (size(theta, 2) - 1);
convHeight = size(theta, 1) - 1;
convWidth = size(theta, 2) - 1;

outputMap = zeros(height, width);
for i = 1:height
  for j = 1:width
    outputMap(i, j) += sum(sum(theta .* inputMap(i:(i + convHeight), j:(j + convWidth))));
  end
end
outputMap += bias;
% ReLU
outputMap = max(0, outputMap);


% tanh activation
%outputImgMat = tanh(outputImgMat);

% softplus
%outputImgMat = log(1 + exp(outputImgMat));

end