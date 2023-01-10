import time
from pathlib import Path

from logic.commons import tio, torch
from logic.loaders import load_test_data_subjects, load_configuration
from logic.preprocessing import get_histogram_landmarks, get_test_transforms
from logic.utils import get_model_and_optimizer, CHANNELS_DIMENSION

def segment_image(input_image, output_path):
    configuration = load_configuration()
    finished = False

    if not configuration:
        return finished

    ###

    # Dataset
    config = {}

    ###
    image_path = [Path(input_image)]

    subjects = load_test_data_subjects(image_path)

    print('Dataset size:', len(tio.SubjectsDataset(subjects)), 'subjects')

    ###

    landmarks = get_histogram_landmarks(image_path, configuration['training']['landmarks_path'])
    config['landmarks'] = landmarks

    ###

    test_transform = get_test_transforms(config)

    test_set = tio.SubjectsDataset(
        subjects, transform=test_transform)

    ###

    device = 'cpu'

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        num_workers=0,
    )

    ###

    model, _ = get_model_and_optimizer(device)
    weights_path = configuration['training']['training_weights_path']

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['weights']) # Epoch final

    ###


    batch = next(iter(test_loader))
    input = batch['mri'][tio.DATA].to(device)
    model.eval()
    FIRST = 0
    FOREGROUND = 1
    with torch.no_grad():
        probabilities = model(input).softmax(dim=CHANNELS_DIMENSION)[:, FOREGROUND:].cpu()
    affine = batch['mri'][tio.AFFINE][FIRST].numpy()

  
    tio.ScalarImage(tensor=torch.round(probabilities[FIRST]), affine=affine).save(output_path+"/"+batch['mri'][tio.STEM][FIRST]+"_unet.nii")
    tio.ScalarImage(tensor=batch['mri'][tio.DATA][FIRST], affine=affine).save(output_path+"/"+batch['mri'][tio.STEM][FIRST]+"_preprocessed.nii")

    for x in range(len(probabilities)):
        print("\033[92m {}\033[00m" .format("The segmentation volume is: "+ str(int(torch.round(probabilities[x]).sum().item())) + " mm^3"))
    
    finished = True

    return finished