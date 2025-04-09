%% Download datasets
% This script will require 'pip install kagglehub'
insert(py.sys.path, int32(0), 'download_data.py');

% Call the Python function
py_result = py.download_data.download_datasets();

% Convert Python dictionary to MATLAB structure
matlab_paths.colour_classification = char(py_result{'colour_classification'});
matlab_paths.license_plate_chars = char(py_result{'license_plate_chars'});
matlab_paths.license_plate_segmentation = char(py_result{'license_plate_segmentation'});
matlab_paths.car_type = char(py_result{'car_type'});

% Assign paths to different data sections
colour_classification_root = matlab_paths.colour_classification;
colour_classification_test = colour_classification_root + "\test";
colour_classification_train = colour_classification_root + "\train";
colour_classification_val = colour_classification_root + "\val";

% This does not have distinct training and testing data, needs to be
% separated.
license_plate_chars_path = matlab_paths.license_plate_chars;

% This does not have distinct training and testing data, needs to be
% separated.
license_plate_segmentation_path = matlab_paths.license_plate_segmentation;

% Assign paths to different data sections
car_type_root = matlab_paths.car_type;
car_type_test = car_type_root + "\car_data\car_data\test";
car_type_train = car_type_root + "\car_data\car_data\train";

% Display the root paths to each data set
disp(['Colour Classification Path: ', colour_classification_root]);
disp(['License Plate Characters Path: ', license_plate_chars_path]);
disp(['License Plate Segmentation Path: ', license_plate_segmentation_path]);
disp(['Car Type Path: ', car_type_root]);

%%  Preprocessing
% Define output directories for preprocessed images
preprocessed_root = fullfile(pwd, 'preprocessed_data');
if ~exist(preprocessed_root, 'dir')
    mkdir(preprocessed_root);
end

% Function to process and store images
function datastoreObj = preprocess_and_store_images(input_path, output_path, convert_gray)
    % Create output folder if it doesn't exist
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % Create an imageDatastore for input images
    imgDS = imageDatastore(input_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % Process each image
    for i = 1:numel(imgDS.Files)
        img = imread(imgDS.Files{i}); % Read image
        img = imresize(img, [227 227]); % Resize image
        
        % Convert to grayscale if required
        if convert_gray && size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % Preserve original filename
        [~, name, ext] = fileparts(imgDS.Files{i});
        output_file = fullfile(output_path, [name, ext]);
        
        % Save processed image
        imwrite(img, output_file);
    end
    
    % Create new imageDatastore with processed images
    datastoreObj = imageDatastore(output_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end
%NOTE: annotation data may need be re-mapped to these processed images. May
%be better to do the mapping before processing, during the function above. Also need to consider
%things like bounding boxes, which will not map nicely to scaled images.
% Process and store images for each dataset
colour_classification_train_DS = preprocess_and_store_images(colour_classification_train, fullfile(preprocessed_root, 'colour_classification_train'), false);
colour_classification_val_DS = preprocess_and_store_images(colour_classification_val, fullfile(preprocessed_root, 'colour_classification_val'), false);
colour_classification_test_DS = preprocess_and_store_images(colour_classification_test, fullfile(preprocessed_root, 'colour_classification_test'), false);

license_plate_chars_DS = preprocess_and_store_images(license_plate_chars_path, fullfile(preprocessed_root, 'license_plate_chars'), true);
license_plate_segmentation_DS = preprocess_and_store_images(license_plate_segmentation_path, fullfile(preprocessed_root, 'license_plate_segmentation'), true);

car_type_train_DS = preprocess_and_store_images(car_type_train, fullfile(preprocessed_root, 'car_type_train'), true);
car_type_test_DS = preprocess_and_store_images(car_type_test, fullfile(preprocessed_root, 'car_type_test'), true);

% Display message indicating preprocessing is complete
disp('Preprocessing complete. Processed imageDatastore objects are ready.');
