function outputMaps = convForward(inputMaps, theta, bias, numIn, numOut)

height = size(inputMaps{1}, 1) - (size(theta{1}, 1) - 1);
width = size(inputMaps{1}, 2) - (size(theta{1}, 2) - 1);
convHeight = size(theta{1}, 1) - 1;
convWidth = size(theta{1}, 2) - 1;

for l = 1:numOut
  outputMaps{l} = zeros(height, width);
  for k = 1:numIn
    for i = 1:height
      for j = 1:width
        outputMaps{l}(i, j) += sum(sum(theta{l} .* inputMaps{k}(i:(i + convHeight), j:(j + convWidth))));
      end
    end
  end
  outputMaps{l} += bias(l);
  % ReLU
  outputMaps{l} = max(0, outputMaps{l});
end

% tanh activation
%outputImgMat = tanh(outputImgMat);

% softplus
%outputImgMat = log(1 + exp(outputImgMat));

end