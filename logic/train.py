
import time
import random
import multiprocessing
from pathlib import Path

from logic.commons import tio, torch
from logic.loaders import load_train_data_subjects, load_configuration
from logic.preprocessing import get_histogram_landmarks, get_training_transforms, get_test_transforms
from logic.utils import get_model_and_optimizer, train


def train_model(images_path, labels_path):
    # Config

    configuration = load_configuration()
    finished = False

    if not configuration:
        return finished

    seed = 42  # for reproducibility
    training_split_ratio = configuration['training']['training_split_ratio']
    num_epochs = configuration['training']['num_epochs']

    ###

    random.seed(seed)
    torch.manual_seed(seed)
    num_workers = multiprocessing.cpu_count()

    ###

    # Dataset
    config = {}

    ###

    images_dir = Path(images_path)
    labels_dir = Path(labels_path)
    image_paths = sorted(images_dir.glob('*.nii.gz'))
    label_paths = sorted(labels_dir.glob('*.nii.gz'))
    assert len(image_paths) == len(label_paths)

    subjects = load_train_data_subjects(image_paths, label_paths)

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(dataset), 'subjects')


    ###

    landmarks = get_histogram_landmarks(image_paths, configuration['training']['landmarks_path'])
    config['landmarks'] = landmarks
    ###

    num_subjects = len(dataset)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)


    training_transform = get_training_transforms(config)

    validation_transform = get_test_transforms(config)


    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')

    ###

    device = 'cpu'


    training_batch_size = 2
    validation_batch_size = 2 * training_batch_size

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        num_workers=0,
    )

    ###

    model, optimizer = get_model_and_optimizer(device)
    weights_path = configuration['training']['training_data_path']
    weights_stem = 'whole_images'
    train_losses, val_losses = train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem, device)
    checkpoint = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'weights': model.state_dict(),
        }
    torch.save(checkpoint, weights_path)

    finished = True

    return finished
    ###