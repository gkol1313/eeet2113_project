close all;clc; clear all;

function data = preprocessData(data,inputSize)
    for ii = 1:size(data,1)
        I = data{ii,1};
        imgSize = size(I);
        bboxes = data{ii,2};
        I = imresize(I,inputSize(1:2));
        scale = inputSize(1:2)./imgSize(1:2);
        bboxes = bboxresize(bboxes,scale);
        data(ii,1:2) = {I,bboxes};
    end
end

function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

    data = cell(size(A));
    for ii = 1:size(A,1)
        I = A{ii,1};
        bboxes = A{ii,2};
        labels = A{ii,3};
        sz = size(I);
    
        % Apply random color jitter.
        I = jitterColorHSV(I,"Brightness",0.4,"Contrast",0.4,"Saturation",0.4, "Hue",0.3);
    
        % Randomly flip image and apply scaling and rotation and shearing
        tform = randomAffine2d(XReflection=true, Scale=[0.9 1.1], XTranslation=[-15 15], YTranslation=[-15 15]);
        rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
        I = imwarp(I,tform,OutputView=rout);
    
        % % Transform bounding boxes
        [bboxes, indices] = bboxwarp(bboxes, tform, rout, 'OverlapThreshold', 0.4);
        labels = labels(indices);
       
        I = imgaussfilt(I, rand() * 0.7); % Random standard deviation up to 0.7
        % Return original data only when all boxes are removed by warping.
        if isempty(indices)
            data(ii,:) = A(ii,:);
        else
            data(ii,:) = {I,bboxes,labels};
        end
    end
end


%% Download datasets
% This script will require 'pip install kagglehub'
insert(py.sys.path, int32(0), 'download_data.py');

% Call the Python function
py_result = py.download_data.download_datasets();

matlab_paths.license_plate_chars = char(py_result{'license_plate_chars'});
matlab_paths.license_plate_segmentation = char(py_result{'license_plate_segmentation'});

license_plate_chars_path = matlab_paths.license_plate_chars;
license_plate_segmentation_path = matlab_paths.license_plate_segmentation;


% Display the root paths to each data set
disp(['License Plate Characters Path: ', license_plate_chars_path]);
disp(['License Plate Segmentation Path: ', license_plate_segmentation_path]);

%% License plate segmentation with filtering
% Define annotation path
annotationPath = license_plate_segmentation_path + "/annotations"; % Folder containing XML files
fullImagePath = license_plate_segmentation_path + "/images";

% Get list of file paths
imageFiles = dir(fullfile(fullImagePath, '*.png'));
annotationFiles = dir(fullfile(annotationPath, '*.xml'));

% Initialize variables
imgPaths = {};
bBoxes = cell(length(annotationFiles), 1); % Store bounding boxes in one cell per row
labels = cell(length(annotationFiles), 1); % Store labels in one cell per row
className = categorical({'licensePlate'});

% Loop through each XML file
for i = 1:length(annotationFiles)
    annotationName = fullfile(annotationPath, annotationFiles(i).name);
    imgName = fullfile(imageFiles(i).folder, imageFiles(i).name);

    % Read XML file
    xmlDoc = xmlread(annotationName);
    bndboxList = xmlDoc.getElementsByTagName('bndbox');

    % Initialize bounding box matrix for this image
    bboxMatrix = zeros(bndboxList.getLength, 4);

    % Loop through each <bndbox> entry
    for j = 0:bndboxList.getLength-1
        bboxNode = bndboxList.item(j);

        % Extract bounding box values
        xmin = str2double(bboxNode.getElementsByTagName('xmin').item(0).getTextContent);
        ymin = str2double(bboxNode.getElementsByTagName('ymin').item(0).getTextContent);
        xmax = str2double(bboxNode.getElementsByTagName('xmax').item(0).getTextContent);
        ymax = str2double(bboxNode.getElementsByTagName('ymax').item(0).getTextContent);

        % Convert to M×4 format: [xmin, ymin, width, height]
        widthBox = xmax - xmin;
        heightBox = ymax - ymin;
        bboxMatrix(j+1, :) = [xmin, ymin, widthBox, heightBox];
    end

    % Store bounding boxes as a cell array
    bBoxes{i} = bboxMatrix;

    % Assign the same label to all bounding boxes
    labels{i} = repmat("licensePlate", size(bboxMatrix, 1), 1);

    imgPaths{i, 1} = imgName;
end

data = table(imgPaths, bBoxes, labels, 'VariableNames', {'imageFilename', 'licensePlate', 'Labels'});
inputSize = [224 224 3];

shuffledIndices = randperm(height(data));
idx = floor(0.7 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = data(shuffledIndices(trainingIdx),:);
trainingData = combine(imageDatastore(trainingDataTbl{:, "imageFilename"}), boxLabelDatastore(trainingDataTbl(:, "licensePlate")));
trainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
boxData = readall(trainingData);
boxData = boxLabelDatastore(table(boxData(:, 2)));

validationIdx = idx+1 : idx + 1 + floor(0.2 * length(shuffledIndices) );
validationDataTbl = data(shuffledIndices(validationIdx),:);
validationData = combine(imageDatastore(validationDataTbl{:, "imageFilename"}), boxLabelDatastore(validationDataTbl(:, "licensePlate")));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = data(shuffledIndices(testIdx),:);
testData = combine(imageDatastore(testDataTbl{:, "imageFilename"}), boxLabelDatastore(testDataTbl(:, "licensePlate")));
testData = transform(testData,@(data)preprocessData(data,inputSize));

annotateTest = read(trainingData);
I = annotateTest{1};
bBox = annotateTest{2};
annotatedImage = insertShape(I,"rectangle",bBox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

reset(trainingData);


maxNumAnchors = 32;
meanIoU = zeros([maxNumAnchors,1]);
anchorBoxes = cell(maxNumAnchors, 1);
for k = 1:maxNumAnchors
    % Estimate anchors and mean IoU.
    [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(boxData,k);
end
figure
plot(1:maxNumAnchors,meanIoU,'-o')
ylabel("Mean IoU")
xlabel("Number of Anchors")
title("Number of Anchors vs. Mean IoU")


augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(8,1);
for k = 1:8
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)
reset(augmentedTrainingData);

if ~exist("trained_detector.mat", 'file') % perform training if file does not exist
    inputSize = [224 224 3];
    baseNet = imagePretrainedNetwork("resnet50");
    imageSize = baseNet.Layers(1).InputSize;
    layerName = baseNet.Layers(1).Name;

    newInputLayer = imageInputLayer(imageSize,Normalization="none",Name=layerName);
    dlnet = replaceLayer(baseNet,layerName,newInputLayer);
    featureExtractionLayers = ["activation_34_relu", "activation_44_relu"];
    className = "licensePlate";

    numAnchors = 6;
    [anchors,meanIoU] = estimateAnchorBoxes(boxData,numAnchors);
    area = anchors(:, 1).*anchors(:,2);
    [~,idx] = sort(area,"descend");

    anchors = anchors(idx,:);
    anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};

    detector = yolov4ObjectDetector(dlnet,className,anchorBoxes, ...
        DetectionNetworkSource=featureExtractionLayers, InputSize=inputSize);%

    % Training options
    options = trainingOptions('adam', ...
    InitialLearnRate=0.00002, ...
    LearnRateSchedule='cyclical', ...
    shuffle= 'every-epoch',...
    LearnRateDropFactor=0.5, ...
    ValidationData=validationData,...
    MiniBatchSize=2, ...
    MaxEpochs=80, ...
    ResetInputNormalization=true, ...
    ValidationPatience=20, ...           
    VerboseFrequency=5);


    plateDetector = trainYOLOv4ObjectDetector(augmentedTrainingData, detector, options);
    % Save trained model
    save('trained_detector.mat',"plateDetector");
else
    plateDetector = load("trained_detector.mat", "plateDetector").plateDetector;
end

data = read(testData);
I = data{1};
[bboxes,scores,labels] = detect(plateDetector, I);
% Apply Non-Maximum Suppression (NMS)

[selectedBboxes,selectedScores,selectedLabels,index] = selectStrongestBboxMulticlass(bboxes,scores,labels,...
    'RatioType','Min','OverlapThreshold',0.65);


I = insertObjectAnnotation(I,"rectangle",selectedBboxes,selectedScores);
figure
imshow(I)

detectionResults = detect(plateDetector,testData,Threshold=0.5);
metrics = evaluateObjectDetection(detectionResults,testData);
AP = averagePrecision(metrics);
[precision,recall] = precisionRecall(metrics,ClassName="licensePlate");

figure
plot(recall{:},precision{:})
xlabel("Recall")
ylabel("Precision")
grid on
title(sprintf("Average Precision = %.2f",AP))


figure
for i = 1:numel(testData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 1}.Files)
    data = read(testData);
    I = data{1};
    [bboxes,scores,labels] = detect(plateDetector, I);
    % Apply Non-Maximum Suppression (NMS)
    
    [selectedBboxes,selectedScores,selectedLabels,index] = selectStrongestBboxMulticlass(bboxes,scores,labels,...
        'RatioType','Min','OverlapThreshold',0.65);
    
    
    I = insertObjectAnnotation(I,"rectangle",selectedBboxes,selectedScores);
    
    imshow(I)
    pause(1);
end