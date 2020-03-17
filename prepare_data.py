import os
import pandas as pd
import numpy as np


def prepare_data():
    path = './data'

    dataset = []

    part_folder_paths = [f.path for f in os.scandir(path) if f.is_dir()]

    for part_path in part_folder_paths:
        file_paths = [f.path for f in os.scandir(part_path) if f.is_file()]

        part = []

        for file_path in file_paths:
            file = open(file_path)

            if 'legit' in file_path:
                # legit
                label = 1
            else:
                # spmsg
                label = 0

            features = np.array(file.read())

            message = np.array([features, label])

            part.append(message)
            file.close()

        dataset.append(part)

    dataset = np.array(dataset)

    return dataset


def split_data(test_part_num, dataset):
    test_features = dataset[test_part_num][:, 0]
    test_labels = dataset[test_part_num][:, 1]

    train_messages = []

    for part_number in range(len(dataset)):
        if part_number != test_part_num:
            for message in dataset[part_number]:
                train_messages.append(message)

    train_messages = np.array(train_messages)

    train_features = train_messages[:, 0]
    train_labels = train_messages[:, 1]

    return train_features, train_labels, test_features, test_labels
