import numpy as np
from keras.utils import to_categorical

from mutation_consumer import MutationConsumer
from util import model_ok


class NeuronContributionMatrix(MutationConsumer):
    def __init__(self, original_model, input_data):
        if not model_ok(original_model):
            raise ValueError('An instance of Sequential or Model is expected')
        self._original_model = original_model
        self._input = input_data
        orig_output = original_model.predict(input_data, verbose=None)
        num_classes = orig_output.shape[-1]
        self._orig_output = to_categorical(np.argmax(orig_output, axis=1), num_classes)
        self._num_classes = num_classes
        # output index -> (layer_index, unit_indices) -> sum of distances from original values at ith output
        self._matrix = dict()

    def consume(self, mutation):
        mutation_output = mutation.get_mutant().predict(self._input, verbose=None)
        mutation_output = to_categorical(np.argmax(mutation_output, axis=1), self._num_classes)
        loc = mutation.get_location()
        li, ui = loc.get_layer_index(), tuple(loc.get_unit_indices())  # convert list to tuple to make it hashable
        diff = np.sum(np.abs(self._orig_output - mutation_output), axis=0)
        if (li, ui) not in self._matrix:
            self._matrix[(li, ui)] = np.zeros(self._num_classes)
        self._matrix[(li, ui)] += diff

    def get_expanded_contrib_matrix(self):
        matrix = dict()
        for oi in range(self._num_classes):
            matrix[oi] = dict()
        for (li, ui), diffs in self._matrix.items():
            for oi in range(self._num_classes):
                if li not in matrix[oi]:
                    matrix[oi][li] = []
                matrix[oi][li].append((ui, diffs[oi]))
        sorted_matrix = {oi: {li: sorted(pl, key=lambda p: p[1], reverse=True)
                              for li, pl in inner_dict.items()} for oi, inner_dict in matrix.items()}
        return sorted_matrix
