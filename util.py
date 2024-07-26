import numpy as np
from keras import Sequential
from keras.utils import to_categorical
import time
import random
import hashlib


def slice_data(inputs, outputs, target):
    return inputs, outputs[:, target]


def binarize_and_balance(inputs, outputs, target):
    dataset_dict = dict()
    rows_count = inputs.shape[0]
    for row_index in range(0, rows_count):
        label = np.argmax(outputs[row_index])
        if label not in dataset_dict:
            dataset_dict[label] = []
        dataset_dict[label].append(inputs[row_index])
    balanced_inputs = dataset_dict[target]
    target_points_sz = len(balanced_inputs)
    balanced_bin_outputs = [1 for _ in range(target_points_sz)]
    num_classes = 2
    for i in range(target_points_sz):
        if len(balanced_inputs) >= num_classes * target_points_sz:
            break
        for label, points in dataset_dict.items():
            if label == target or i >= len(points):
                continue
            balanced_inputs.append(points[i])
            balanced_bin_outputs.append(0)
    return np.asarray(balanced_inputs), to_categorical(np.asarray(balanced_bin_outputs), num_classes)


def model_ok(model):
    return isinstance(model, Sequential)


def jaccard_sim(set_a, set_b):
    intersection = len(set(set_a).intersection(set_b))
    union = (len(set_a) + len(set_b)) - intersection
    return float(intersection) / union


def make_axes(w):
    return tuple(range(len(w.shape) - 1))


class MutationUIDGenerator:
    def __init__(self):
        self.generated_uids = set()

    def generate(self):
        while True:
            current_time = time.time_ns()
            random_number = random.randint(0, 999999)
            unique_string = f"{current_time}{random_number}"
            unique_number = int(hashlib.sha256(unique_string.encode()).hexdigest(), 16)
            if unique_number not in self.generated_uids:
                self.generated_uids.add(unique_number)
                return unique_number
