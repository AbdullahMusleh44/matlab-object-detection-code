clc; 
clear;

% Setting doTraining variable to true
doTraining = true;

% Loading the custom dataset
GTruthData = load('PERSONS225DS.mat');
source = GTruthData.gTruth.DataSource.Source;
labelData =  GTruthData.gTruth.LabelData.Person;
personDataset = table(source, labelData);

% Display first 6 rows personDataset.
personDataset(1:6,:)

% Setting rng to 0, shuffle the random indices of personDataset into
% training validation and testing sets
rng(0);
shuffledIndices = randperm(height(personDataset));
idx = floor(0.6 * height(personDataset));

% Indicies of training set
trainingIdx = 1:idx;
trainingDataTbl = personDataset(shuffledIndices(trainingIdx),:);

% Getting the indicies of validation set 10%
validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = personDataset(shuffledIndices(validationIdx),:);

% Getting indices for test set 
testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = personDataset(shuffledIndices(testIdx),:);

% Creating image datastore and box label datastore for training set
imdsTrain = imageDatastore(trainingDataTbl{:,'source'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'labelData'));

% Creating image datastore and box label datastore for validation set
imdsValidation = imageDatastore(validationDataTbl{:,'source'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'labelData'));

% Creating image datastore and box label datastore for test set
imdsTest = imageDatastore(testDataTbl{:,'source'});
bldsTest = boxLabelDatastore(testDataTbl(:,'labelData'));

% Combine image/box label datastores to form training validation and test
% datasets
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% Creating Faster R-CNN network
inputSize = [224 224 3];

% Preprocess training data and estimate the anchor boxes 
preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors);

% Defining feature extractio network, feature layer and number of classes
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
numClasses = width(personDataset)-1;

% Faster R-CNN layer graph
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% Use augment data function
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

% Pre processing augmented training and validation data with preprocess
% function
trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

% Display one preprocessed training image
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Training Faster R-CNN with custom hyperparameters
options = trainingOptions('sgdm',...
    'MaxEpochs',3,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'ValidationData',validationData);

if doTraining
    % Train the Faster R-CNN detector.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
else
    % Load the pretrained detector from the example if needed.
    detector = pretrained.detector;
end

% Read one test image and resize to input size
I = imread(testDataTbl.source{5});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

% Insert bounding box with score on image
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% Load test data
GTruthData2 = load('PERSONS40DS.mat');
source = GTruthData2.gTruth.DataSource.Source;
labelData =  GTruthData2.gTruth.LabelData.Person;
Dataset2 = table(source, labelData);

% Image datastore
imageDS2 = imageDatastore('PERSONS45DSresized\');
% label datastore
boxLabelDS2 = boxLabelDatastore(Dataset2(:,'labelData'));
% Combine the image and label datastores
testingDs = combine(imageDS2,boxLabelDS2);

% Detect in testing dataset 
detectionResults = detect(detector,imageDS2,'MinibatchSize',4); 

% Evaluation, get average precision, recall, miss rate, fppi etc
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,boxLabelDS2);

% Plot precision-recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

% Plot log-average miss rate vs false positives
[am, fppi, missRate] = evaluateDetectionMissRate(detectionResults, boxLabelDS2);
figure
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.1f', am))



