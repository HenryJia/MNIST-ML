function outputMaps = maxPool(x, poolHeight, poolWidth, numIn)
% note poolHeight and poolWidth must be factors of the dimensions of X

numHeight = size(x{1}, 1) / poolHeight;
numWidth = size(x{1}, 2) / poolWidth;

for k = 1:numIn
  outputMaps{k} = zeros(numHeight, numWidth);
  for i = 1:numHeight
    for j = 1:numWidth
      %sample = x((i * poolHeight):((i + 1) * poolHeight - 1), (j * poolWidth):((j + 1) * poolWidth - 1));
      sample = x{k}(((i - 1) * poolHeight + 1):(i * poolHeight), ((j - 1) * poolWidth + 1):(j * poolWidth));
      outputMaps{k}(i, j) = max(max(sample));
    end
  end
end

end