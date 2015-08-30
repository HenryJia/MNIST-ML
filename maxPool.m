function outputMap = maxPool(inputMap, poolHeight, poolWidth)
% note poolHeight and poolWidth must be factors of the dimensions of X

numHeight = size(inputMap, 1) / poolHeight;
numWidth = size(inputMap, 2) / poolWidth;

outputMap = zeros(numHeight, numWidth);
for i = 1:numHeight
  for j = 1:numWidth
    sample = inputMap(((i - 1) * poolHeight + 1):(i * poolHeight), ((j - 1) * poolWidth + 1):(j * poolWidth));
    outputMap(i, j) = max(max(sample));
  end
end

end