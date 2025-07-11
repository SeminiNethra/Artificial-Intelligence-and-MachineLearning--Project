clear; clc;
load('combinedFeatures.mat', 'featureMatrix', 'labels');
corrThreshold = 0.2;
correlations = abs(corr(featureMatrix, labels)); 
selectedFeatures = find(correlations > corrThreshold);
disp('Selected Features:');
disp(selectedFeatures);
reducedFeatureMatrix = featureMatrix(:, selectedFeatures);
splitRatio = 0.8;
rng(42);
numSamples = size(reducedFeatureMatrix, 1);
indices = randperm(numSamples);
splitIndex = round(splitRatio * numSamples);
trainIndices = indices(1:splitIndex);
testIndices = indices(splitIndex+1:end);
XTrain = reducedFeatureMatrix(trainIndices, :);
yTrain = labels(trainIndices, :);
XTest = reducedFeatureMatrix(testIndices, :);
yTest = labels(testIndices, :);
hiddenLayerConfigs = {[16], [32]};
learningRates = [0.01];
numEpochs = [50]; 

bestNet = [];
bestAccuracy = 0;

for hConfig = hiddenLayerConfigs
    for lr = learningRates
        for epochs = numEpochs
            net = feedforwardnet(hConfig{1});
            net.trainParam.lr = lr;
            net.trainParam.epochs = epochs;
            net.trainParam.goal = 1e-4; % Early stopping goal
            net.divideFcn = 'dividerand';
            net.divideParam.trainRatio = 0.7;
            net.divideParam.valRatio = 0.3;
            net.divideParam.testRatio = 0.3;
            [XTrainTransposed, yTrainOneHot] = preprocessData(XTrain, yTrain, max(yTrain));
            try
                [net, tr] = train(net, XTrainTransposed, yTrainOneHot);
                XTestTransposed = XTest';
                yPredOneHot = net(XTestTransposed);
                [~, yPred] = max(yPredOneHot, [], 1);
                accuracy = sum(yPred' == yTest) / length(yTest) * 100;

                % Update best model
                if accuracy > bestAccuracy
                    bestAccuracy = accuracy;
                    bestNet = net;
                end
            catch ME
                disp(['Error during training: ', ME.message]);
            end
        end
    end
end
disp(['Best Accuracy: ', num2str(bestAccuracy), '%']);
save('optimizedNetwork.mat', 'bestNet', 'bestAccuracy');
function [XTransposed, yOneHot] = preprocessData(X, y, numClasses)
    % Transpose the feature matrix
    XTransposed = X';   
    yOneHot = zeros(numClasses, length(y));
    for i = 1:length(y)
        yOneHot(y(i), i) = 1;
    end
end