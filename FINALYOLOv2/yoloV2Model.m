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

% Setting rng to 0, shuffle the indices of personDataset and then reorder
rng(0);
shuffledIdx = randperm(height(personDataset));
trainingData = personDataset(shuffledIdx,:);

% Creating image and label datastores
imageDS = imageDatastore(trainingData.source);
boxLabelDS = boxLabelDatastore(trainingData(:,2:end));

% Combining the two datastores
cds = combine(imageDS, boxLabelDS);

% Define input size and number of classes for object detection
inputSize = [224 224 3];
numClasses = width(personDataset)-1;

% Preprocess the data for training, resize and normalize
trainingDataForEstimation = transform(cds,@(data)preprocessData(data,inputSize));

% Estimate anchor boxes for detection
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

% Set feature extraction network and feature layer to use
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

% Define YOLOv2 network architecture
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% Augment training data
augmentedTrainingData = transform(cds,@augmentData);

% Visualize the augmented training data
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

% Preprocess the augemnted training data, resize and normalize 
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));

% Read data, resize and detect, display results
data = read(preprocessedTrainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Custom YOLOv2 Object Detector Hyperparameters
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-2, ...
        'MaxEpochs',20, ... 
        'CheckpointPath',tempdir);

if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    detector = yolov2ObjectDetector('darknet19-coco');
end


% Load test data
GTruthData2 = load('PERSONS40DS.mat');
source = GTruthData2.gTruth.DataSource.Source;
labelData =  GTruthData2.gTruth.LabelData.Person;
Dataset2 = table(source, labelData);

% Image and Label datastores
imageDS2 = imageDatastore(source);
boxLabelDS2 = boxLabelDatastore(Dataset2(:,'labelData'));
% Combine the image and label datastores
testingDs = combine(imageDS2,boxLabelDS2);

% Detect in testing dataset
detectionResults = detect(detector, imageDS2,Threshold=0.001);

% Evaluation, get average precision, recall, miss rate, fppi etc
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, boxLabelDS2);

% Plot precision-recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

% Plot log-average miss rate vs false positives
[am, fppi, missRate] = evaluateDetectionMissRate(detectionResults, boxLabelDS2);
figure
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.1f', am))
