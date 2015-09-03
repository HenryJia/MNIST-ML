function convolvedFeatures = cnnConvolve(filterDim, numOutputs, inputMaps, W, b)

%cnnConvolve Returns the convolution of the features given by W and b with
%the given inputMaps
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  inputMaps - large inputMaps to convolve with, matrix in the form
%           inputMaps(r, c, inputMap number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numInputMaps = size(inputMaps, 3);
imageDim = size(inputMaps, 1);
convDim = imageDim - filterDim + 1;

%convolvedFeatures = zeros(convDim, convDim, numFilters, numinputMaps);
convolvedFeatures = zeros(convDim, convDim, numOutputs);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numinputMaps
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 inputMaps should take less than 30 seconds 
%   Convolving with 5000 inputMaps should take around 2 minutes
%   (So to save time when testing, you should convolve with less inputMaps, as
%   described earlier)


for outputNum = 1:numOutputs
  % convolution of image with feature matrix
  convolvedImage = zeros(convDim, convDim);
  for inputNum = 1:numInputMaps

    % Obtain the feature (filterDim x filterDim) needed during the convolution

    %%% YOUR CODE HERE %%%
    filter = W(:,:, inputNum, outputNum);

    % Flip the feature matrix because of the definition of convolution, as explained later
    filter = rot90(squeeze(filter), 2);
      
    % Obtain the image
    im = squeeze(inputMaps(:, :, inputNum));

    % Convolve "filter" with "im", adding the result to convolvedImage
    % be sure to do a 'valid' convolution

    %%% YOUR CODE HERE %%%
    convolvedImage = convolvedImage + conv2(im, filter, 'valid');
    
    % Add the bias unit
    % Then, apply the sigmoid function to get the hidden activation

    %%% YOUR CODE HERE %%%
    
  end
  convolvedImage = convolvedImage + b(outputNum);
  %convolvedImage = sigmoid(convolvedImage);
  convolvedFeatures(:, :, outputNum) = convolvedImage;
end


end