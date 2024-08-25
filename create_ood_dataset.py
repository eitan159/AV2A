import pandas as pd
import json


llp_classes = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']

llp_classes = [c.lower().replace("_", " ") for c in llp_classes]

with open("/cortex/data/video/AVE/Annotations.txt", 'r') as f:
    ave_data = f.read().strip().split("\n")[1:]

ave_classes = list({sample.split("&")[0].lower() for sample in ave_data})

intsection_classes = (set(ave_classes) & set(llp_classes)).union({'baby cry, infant cry', 'violin, fiddle'})

ave_class2llp_class = {
    'baby cry, infant cry': 'Baby_cry_infant_cry',
    'violin, fiddle': 'Violin_fiddle',
    'banjo': 'Banjo',
    'helicopter': 'Helicopter',
    'motorcycle': 'Motorcycle',
    'frying (food)': 'Frying_(food)',
    'acoustic guitar': 'Acoustic_guitar',
    'cat': 'Cat',
    'accordion': 'Accordion',
    'chainsaw': 'Chainsaw'
}


ave_files_intescted_classes = []
for sample in ave_data:
    sample = sample.split("&")
    if sample[0].replace("_", " ").lower() in intsection_classes:
        ave_files_intescted_classes.append(sample[1])

df = pd.read_csv("/cortex/data/video/LLP/AVVP_dataset_full.csv", header=0, sep='\t')
llp_files = ['_'.join(file_name.split('_')[:-2]) for file_name in list(df["filename"].values)]


ave_domain_shift_filenames = list(set(ave_files_intescted_classes) - set(llp_files))
class_count = {c:0 for c in intsection_classes}
final_filenames = {}
for sample in ave_data:
    sample = sample.split("&")
    if sample[1] in ave_domain_shift_filenames and sample[0].replace("_", " ").lower() in intsection_classes:
        if sample[1] not in final_filenames:
            final_filenames[sample[1]] = []
        class_count[sample[0].replace("_", " ").lower()] += 1
        final_filenames[sample[1]].append({'class': sample[0].replace("_", " ").lower(),
                                           'llp_class': ave_class2llp_class[sample[0].replace("_", " ").lower()],
                                           'start': int(sample[3]),
                                           'end': int(sample[4])})

# final_filenames = list(set(final_filenames))

with open("ood_dataset.json", "w") as f:
    json.dump(final_filenames, f)