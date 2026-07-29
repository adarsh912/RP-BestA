import pandas as pd

UCR_CATALOG = [
    ('GunPoint', 'Motion', 50, 150, 150, 2),
    ('Coffee', 'Spectro', 28, 28, 286, 2),
    ('ArrowHead', 'Image', 36, 175, 251, 3),
    ('ECG200', 'ECG', 100, 100, 96, 2),
    ('Chinatown', 'Sensor', 20, 345, 24, 2),
    ('ItalyPowerDemand', 'Sensor', 67, 1029, 24, 2),
    ('SonyAIBORobotSurface1', 'Sensor', 20, 601, 70, 2),
    ('TwoLeadECG', 'ECG', 23, 1139, 82, 2),
    ('ECGFiveDays', 'ECG', 23, 861, 136, 2),
    ('MoteStrain', 'Sensor', 20, 1252, 84, 2),
    ('Beef', 'Spectro', 30, 30, 470, 5),
    ('OliveOil', 'Spectro', 30, 30, 570, 4),
    ('Meat', 'Spectro', 60, 60, 448, 3),
    ('BeetleFly', 'Image', 20, 20, 512, 2),
    ('BirdChicken', 'Image', 20, 20, 512, 2),
    ('FaceFour', 'Image', 24, 88, 350, 4),
    ('SyntheticControl', 'Simulated', 300, 300, 60, 6),
    ('CBF', 'Simulated', 30, 900, 128, 3),
    ('TwoPatterns', 'Simulated', 1000, 4000, 128, 4),
    ('Wafer', 'Sensor', 1000, 6164, 152, 2),
    ('FordA', 'Sensor', 3601, 1320, 500, 2),
    ('Yoga', 'Image', 300, 3000, 426, 2),
    ('SwedishLeaf', 'Image', 500, 625, 128, 15),
]

def get_catalog():
    return [{'name': c[0], 'domain': c[1], 'train_size': c[2], 'test_size': c[3],
             'length': c[4], 'n_classes': c[5]} for c in UCR_CATALOG]

def get_dataset_names(domains=None, max_length=None, max_train_size=None):
    names = []
    for c in UCR_CATALOG:
        name, domain, train_size, _, length, _ = c
        if domains and domain not in domains:
            continue
        if max_length and length > max_length:
            continue
        if max_train_size and train_size > max_train_size:
            continue
        names.append(name)
    return names

def get_catalog_dataframe():
    return pd.DataFrame(get_catalog())
