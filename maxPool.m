function [outputMap, pos] = maxPool(inputMap, poolHeight, poolWidth)
% note poolHeight and poolWidth must be factors of the dimensions of X

numHeight = size(inputMap, 1) / poolHeight;
numWidth = size(inputMap, 2) / poolWidth;

pos = zeros(2, numHeight, numWidth);
colMax = zeros(1, numWidth);
colPos = zeros(poolHeight, poolWidth);
outputMap = zeros(numHeight, numWidth);

for i = 1:numHeight
  for j = 1:numWidth
    sample = inputMap(((i - 1) * poolHeight + 1):(i * poolHeight), ((j - 1) * poolWidth + 1):(j * poolWidth));
    [colMax, colPos] = max(sample);
    [outputMap(i, j), pos(2, i, j)] = max(colMax);
    pos(1, i, j) = colPos(1, pos(2, i, j));
  end
end

end