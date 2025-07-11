load('normalizedFeatures.mat', 'normalizedFeatures', 'labels');
splitRatio = 0.8;
numSamples = size(normalizedFeatures, 1);
rng(42);
indices = randperm(numSamples);
splitIndex = round(splitRatio * numSamples);
trainIndices = indices(1:splitIndex);
testIndices = indices(splitIndex+1:end);
XTrain = normalizedFeatures(trainIndices, :);
yTrain = labels(trainIndices, :);
XTest = normalizedFeatures(testIndices, :);
yTest = labels(testIndices, :);
disp('Training Set Size:');
disp(size(XTrain)); 
disp(size(yTrain));
disp('Testing Set Size:');
disp(size(XTest)); 
disp(size(yTest)); 
save('splitDataset.mat','XTrain', 'yTrain', 'XTest', 'yTest');

load('normalizedFeatures.mat', 'normalizedFeatures', 'labels');


splitRatio = 0.8;
numSamples = size(normalizedFeatures, 1);
rng(42);  % For reproducibility
indices = randperm(numSamples);
splitIndex = round(splitRatio * numSamples);
trainIndices = indices(1:splitIndex);
testIndices = indices(splitIndex+1:end);
XTrain = normalizedFeatures(trainIndices, :);
yTrain = labels(trainIndices, :);
XTest = normalizedFeatures(testIndices, :);
yTest = labels(testIndices, :);


disp('Training Set Size:');
disp(size(XTrain)); 
disp(size(yTrain));
disp('Testing Set Size:');
disp(size(XTest)); 
disp(size(yTest)); 

save('splitDataset.mat', 'XTrain', 'yTrain', 'XTest', 'yTest');




trainSize = length(trainIndices);
testSize = length(testIndices);

figure;
pie([trainSize, testSize], {'Training Set', 'Testing Set'});
title('Dataset Split: Training vs. Testing');
figure;
bar([1, 2], [trainSize, testSize], 'FaceColor', [0.2, 0.6, 0.2]);
xticks([1, 2]);
xticklabels({'Training Set', 'Testing Set'});
ylabel('Number of Samples');
title('Dataset Split: Training vs. Testing');
grid on;
load('splitDataset.mat', 'XTrain', 'yTrain', 'XTest', 'yTest');
hiddenLayerSizes = [10,30,50]; % Two hidden layers with 128 and 64 neurons
net = feedforwardnet(hiddenLayerSizes);
net.layers{1}.transferFcn = 'tansig'; % Hyperbolic tangent sigmoid for first hidden layer
net.layers{2}.transferFcn = 'tansig'; % Hyperbolic tangent sigmoid for second hidden layer
net.layers{3}.transferFcn = 'softmax'; % Softmax for output layer (classification)

net.trainParam.lr = 0.001; % Learning rate
net.trainParam.epochs = 50; % Number of epochs
net.trainParam.goal = 1e-6; % Training goal (mean squared error)
net.trainParam.min_grad = 1e-6; % Minimum gradient
net.trainParam.max_fail = 10; % Maximum validation failures

net.divideFcn = 'dividerand'; % Randomly divide data into training, validation, and testing
net.divideParam.trainRatio = 0.7; % 80% for training
net.divideParam.valRatio = 0.3; % 20% for validation
net.divideParam.testRatio = 0.3; % Testing is done manually

[XTrainTransposed, yTrainOneHot] = preprocessData(XTrain, yTrain, max(yTrain));

[net, tr] = train(net, XTrainTransposed, yTrainOneHot);

XTestTransposed = XTest'; % Transpose test features for MATLAB compatibility
yPredOneHot = net(XTestTransposed); % Predict outputs
[~, yPred] = max(yPredOneHot, [], 1); % Convert one-hot predictions to class labels

accuracy = sum(yPred' == yTest) / length(yTest) * 100;
disp(['Test Accuracy: ', num2str(accuracy), '%']);

function [XTransposed, yOneHot] = preprocessData(X, y, numClasses)
    % Transpose the feature matrix
    XTransposed = X';
    
    % Convert labels to one-hot encoding
    yOneHot = zeros(numClasses, length(y));
    for i = 1:length(y)
        yOneHot(y(i), i) = 1;
    end
end