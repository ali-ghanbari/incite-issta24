from os import path
import os
import shutil
import numpy as np
from keras.models import load_model
from tensorflow import reduce_mean
import pickle
from sklearn.metrics import mean_absolute_error

from contrib_matrix import NeuronContributionMatrix
from mutation_consumer import MutationCounter
from mutation_source import MutationGenerator
from slicer import OptimizedRankedSlicerCls, OptimizedRankedSlicerReg
from unit_clusterer import UnitClustering
from util import binarize_and_balance, jaccard_sim, slice_data


class Entry:
    def __init__(self):
        self._model_filename = None
        self._model = None
        self._train_in = None
        self._train_out = None
        self._test_in = None
        self._test_out = None
        self._validation_split = 0.1
        self._fraction = 0.1
        self._cluster_sz = 10
        self._threshold = 0.6
        self._optimizer = 'adam'
        self._epochs = 64
        self._task_type = 'class'
        self._patience = -1

    def load_model(self, model_filename):
        self._model_filename = model_filename
        self._model = load_model(model_filename)

    def load_train_inputs(self, train_inputs_filename):
        self._train_in = np.load(train_inputs_filename)

    def load_train_outputs(self, train_outputs_filename):
        self._train_out = np.load(train_outputs_filename)

    def load_test_inputs(self, test_inputs_filename):
        self._test_in = np.load(test_inputs_filename)

    def load_test_outputs(self, test_outputs_filename):
        self._test_out = np.load(test_outputs_filename)

    def set_mutation_weight_fraction(self, fraction):
        self._fraction = fraction

    def set_validation_split(self, validation_split):
        self._validation_split = validation_split

    def set_cluster_size(self, cluster_size):
        self._cluster_sz = cluster_size

    def set_cluster_selection_threshold(self, threshold):
        self._threshold = threshold

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_task_type(self, task_type):
        self._task_type = task_type

    def set_patience(self, patience):
        self._patience = patience

    def __get_outputs_count(self):
        return self._model.layers[-1].units

    def run(self):
        counter = MutationCounter()
        cm_filename = 'matrix-optimized-%d-%.2f-%.2f.cm' % (self._cluster_sz,
                                                            self._threshold,
                                                            self._fraction)
        if path.exists(cm_filename):
            print('Skipped mutation analysis, loading %s...' % cm_filename)
            with open(cm_filename, 'rb') as f:
                expanded_contrib_matrix = pickle.load(f)
        else:
            unit_clustering = UnitClustering(self._model)
            clusters = unit_clustering.get_clusters(self._cluster_sz)
            ncm = NeuronContributionMatrix(self._model, self._train_in, class_tt=(self._task_type == 'class'))
            mg = MutationGenerator(self._model_filename, clusters, self._fraction)
            mg.get_mutations([ncm, counter])
            expanded_contrib_matrix = ncm.get_expanded_contrib_matrix()
            with open(cm_filename, 'wb') as f:
                pickle.dump(expanded_contrib_matrix, f)
        counter.reset()
        out_dir = 'modules-optimized-%d-%.2f-%.2f' % (self._cluster_sz,
                                                      self._threshold,
                                                      self._fraction)
        if path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        original_predictions = self._model.predict(self._test_in, verbose=None)
        selected_units = dict()
        for target in range(self.__get_outputs_count()):
            module_name = 'module-%d' % target
            if self._task_type == 'class':
                bin_train_in, bin_train_out = binarize_and_balance(self._train_in, self._train_out, target)
                bin_test_in, bin_test_out = binarize_and_balance(self._test_in, self._test_out, target)
                slicer = OptimizedRankedSlicerCls(self._model_filename,
                                                  expanded_contrib_matrix,
                                                  self._threshold,
                                                  train_data=(bin_train_in, bin_train_out),
                                                  validation_split=self._validation_split,
                                                  optimizer=self._optimizer,
                                                  epochs=self._epochs,
                                                  patience=self._patience)
                sliced_model = slicer.get_slice(target)
                with open(path.join(out_dir, module_name + '-perf.txt'), 'w') as f:
                    class_indices = np.where(np.argmax(self._test_out, axis=1) == target)[0]
                    original_class_predictions = np.argmax(original_predictions[class_indices], axis=1)
                    class_true_labels = np.argmax(self._test_out[class_indices])
                    original_model_acc = np.sum(original_class_predictions == class_true_labels) / len(class_indices)
                    module_acc = sliced_model.evaluate(bin_test_in, bin_test_out, verbose=None)[1]
                    f.write('Module Accuracy: %f\n' % module_acc)
                    f.write('Original Model Accuracy: %f\n' % original_model_acc)
            else:
                sliced_train_in, sliced_train_out = slice_data(self._train_in, self._train_out, target)
                sliced_test_in, sliced_test_out = slice_data(self._test_in, self._test_out, target)
                slicer = OptimizedRankedSlicerReg(self._model_filename,
                                                  expanded_contrib_matrix,
                                                  self._threshold,
                                                  train_data=(sliced_train_in, sliced_train_out),
                                                  validation_split=self._validation_split,
                                                  optimizer=self._optimizer,
                                                  epochs=self._epochs,
                                                  patience=self._patience)
                sliced_model = slicer.get_slice(target)
                with open(path.join(out_dir, module_name + '-perf.txt'), 'w') as f:
                    original_model_mae = mean_absolute_error(original_predictions[:, target], sliced_test_out)
                    original_model_mae = reduce_mean(original_model_mae).numpy()
                    module_predictions = sliced_model.predict(sliced_test_in, verbose=None)
                    module_mae = mean_absolute_error(module_predictions, sliced_test_out)
                    module_mae = reduce_mean(module_mae).numpy()
                    f.write('Module MAE: %f\n' % module_mae)
                    f.write('Original Model MAE: %f\n' % original_model_mae)
            selected_units[target] = slicer.get_selected_units()
            sliced_model.save(path.join(out_dir, module_name + '.keras'))

        with open(path.join(out_dir, 'stats.txt'), 'w') as f:
            f.write(str(counter))
            f.write('==============\n')
            f.write('Jaccard similarities:\n')
            count = 0
            ji_sum = 0
            for m1 in range(self.__get_outputs_count()):
                for m2 in range(m1 + 1, self.__get_outputs_count()):
                    ji = jaccard_sim(selected_units[m1], selected_units[m2])
                    f.write('Module %d vs. Module %d: %f\n' % (m1, m2, ji))
                    ji_sum += ji
                    count += 1
            f.write('--------------\n')
            f.write('Average: %f\n' % (ji_sum / count))

