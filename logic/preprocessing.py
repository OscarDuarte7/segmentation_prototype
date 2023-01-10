from logic.commons import np, tio

def get_histogram_landmarks(image_paths, landmarks_path):
    histogram_landmarks_path = landmarks_path

    try:
        landmarks = np.load(histogram_landmarks_path)

    except Exception as ex:    
        landmarks = tio.HistogramStandardization.train(
            image_paths,
            output_path=histogram_landmarks_path,
        )
        print("Generating Histogram Standardization weights")

    return landmarks

def get_training_transforms(config):
    training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad((224, 384, 352)),
    tio.Resample(1),
    tio.CropOrPad((96, 192, 160)),
    tio.HistogramStandardization({'mri': config['landmarks']}),
    tio.OneHot(),
    ])

    return training_transform

def get_test_transforms(config):
    test_transform = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad((224, 384, 352)),
    tio.Resample(1),
    tio.CropOrPad((96, 192, 160)),
    tio.HistogramStandardization({'mri': config['landmarks']}),
    tio.OneHot(),
    ]) 

    return test_transform