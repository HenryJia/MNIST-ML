function outputImgMat = convForward(inputImgMat, theta, bias)

height = size(inputImgMat, 1) - (size(theta, 1) - 1);
width = size(inputImgMat, 2) - (size(theta, 2) - 1);

outputImgMat = zeros(height, width);

for i = 1:height
  for j = 1:width
  outputImgMat(i, j) = sum(sum(theta .* inputImgMat(i:(i + 2), j:(j + 2))));
  end
end

outputImgMat += bias;

% tanh activation
%outputImgMat = tanh(outputImgMat);

% RELU
outputImgMat = log(1 + exp(outputImgMat));

end