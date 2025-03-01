import torch
import random
import numpy as np
import json

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_data(dataset_name):
    if dataset_name == "LLP":
        labels = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                    'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                    'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                    'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                    'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                    'Clapping']
        subset = None

    elif dataset_name == "AVE":
        with open("./test_AVE.json", 'r') as f:
            subset = json.load(f)

        labels = []
        for k, v in subset.items():
            labels.extend([sample['class'] for sample in v])
        
        labels = list(set(labels))

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

    return subset, labels