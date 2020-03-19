import os
import numpy as np


# считываем данные из всех файлов в многомерный массив
def prepare_data():
    dataset = []
    subdirectories = [f.path for f in os.scandir('./data') if f.is_dir()]
    for subdirectory in subdirectories:
        files = [f.path for f in os.scandir(subdirectory) if f.is_file()]
        messages = []
        for file_path in files:
            file = open(file_path)
            features = np.array(file.read())
            # spam messages contain the word 'spmsg' in the title, not spam ones - 'legit'
            message = np.array([features, 1 if 'legit' in file_path else 0])
            messages.append(message)
            file.close()
        dataset.append(messages)
    return np.array(dataset)


# разделяем данные - что под номером test_part_num - будет тестовое, все остальные данные - трейновые
def split_data(test_part_num, dataset):
    train_messages = []
    for part_number in range(len(dataset)):
        if part_number != test_part_num:
            for message in dataset[part_number]:
                train_messages.append(message)
    train_messages = np.array(train_messages)
    # train features, labels, test features, labels
    return train_messages[:, 0], train_messages[:, 1], dataset[test_part_num][:, 0], dataset[test_part_num][:, 1]
