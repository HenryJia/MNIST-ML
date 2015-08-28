function outputImgMat = maxPool(x, poolHeight, poolWidth)
% note poolHeight and poolWidth must be factors of the dimensions of X

numHeight = size(x, 1) / poolHeight;
numWidth = size(x, 2) / poolWidth;

output = zeros(numHeight, numWidth);

for i = 1:numHeight
  for j = 1:numWidth
    %sample = x((i * poolHeight):((i + 1) * poolHeight - 1), (j * poolWidth):((j + 1) * poolWidth - 1));
    sample = x(((i - 1) * poolHeight + 1):(i * poolHeight - 1), ((j - 1) * poolWidth + 1):(j * poolWidth - 1));
    outputImgMat(i, j) = max(max(sample));
  end
end

end