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

% Creating SSD Object Detection Network, defining input size of imgs
inputSize = [300 300 3];

% Defining object classes to detect
classNames = {'Person'};

% Number of object classes to detect
numClasses = width(personDataset)-1;

% Creating SSD network with resnet50 feature extraction
lgraph = ssdLayers(inputSize, numClasses, 'resnet50');

% Data augmentation on the combined datastore
augmentedTrainingData = transform(cds,@augmentData);

% Preprocess the augmented training data to prepare for training
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);

% Custom SSD Object Detector Hyperparameters
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ....
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 20, ...
    'VerboseFrequency', 50, ...
    'CheckpointPath', tempdir, ...
    'Shuffle','every-epoch');

if doTraining
    % Train the SSD detector with the custom data.
    [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);
else
   % Load the pretrained detector from the example if needed.
     detector = pretrained.detector
end

% Read data from combined ds, resize and detect, display results
data = read(cds);
I = data{1,1};
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I, 'Threshold', 0.4);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% Loading test data
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
detectionResults = detect(detector, imageDS2, 'Threshold', 0.01);

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
