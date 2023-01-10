from logic.commons import tio
from logic.commons import yaml
from logic.commons import Schema, And, Use, SchemaError, SchemaMissingKeyError

def load_train_data_subjects(image_paths, label_paths):
    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        segmentation=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return subjects


def load_test_data_subjects(image_paths):
    subjects = []
    for image_path in image_paths:
        subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        )
        subjects.append(subject)
    return subjects

def load_configuration():
    try:
        with open('config/config.yaml') as f:
            
            data = yaml.load(f, Loader=yaml.FullLoader)

            model_schema = Schema({
                                'out_classes': And(Use(int), lambda n: 0 < n),
                                'dimensions': And(Use(int), lambda n: 0 < n < 4),
                                'num_encoding_blocks': And(Use(int), lambda n: 1 < n ),
                                'out_channels_first_layer': And(Use(int), lambda n: 0 < n ),
                                'activation': And(str)       
                            }, ignore_extra_keys=True)

            training_schema = Schema({
                                'num_epochs': And(Use(int), lambda n: 0 < n),
                                'training_split_ratio': And(Use(float), lambda n: 0 < n < 1),
                                'training_data_path': And(str),
                                'training_weights_path': And(str),
                                'landmarks_path': And(str),   
                            }, ignore_extra_keys=True)

            model_schema.validate(data['model'])
            training_schema.validate(data['training'])

        return data
        
    except FileNotFoundError:
        print("\033[91m {}\033[00m" .format("Configuration file not found in config/config.yaml"))
        return None

    except (KeyError,SchemaError,SchemaMissingKeyError):
        print("\033[91m {}\033[00m" .format("Configuration file validation failed please fix config file and try again"))
        return None