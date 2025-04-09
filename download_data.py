import kagglehub

def download_datasets():
    # Colour recognition isnt annotated - determined by directory name, e.g. red
    colour_recognition_path = kagglehub.dataset_download("landrykezebou/vcor-vehicle-color-recognition-dataset")
    # .xml annotations
    license_plate_chars_path = kagglehub.dataset_download("aladdinss/license-plate-annotated-image-dataset")
    license_plate_segmentation_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
    # .csv annotations
    car_type_path = kagglehub.dataset_download("jutrera/stanford-car-dataset-by-classes-folder")

    return {'colour_classification': colour_recognition_path,
            'license_plate_chars': license_plate_chars_path,
            'license_plate_segmentation': license_plate_segmentation_path,
            'car_type': car_type_path}
