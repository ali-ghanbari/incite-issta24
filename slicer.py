import os
import tempfile

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


def create_like_pruned_layer(layer, units=None):
    layer_config = layer.get_config()
    if isinstance(layer, Dense):
        if units is None:
            raise ValueError('Unit must be a positive number')
        layer_config['units'] = units
        layer_config['name'] += '_pruned'
        return Dense.from_config(layer_config)
    elif isinstance(layer, Conv2D):
        if units is None:
            raise ValueError('Unit must be a positive number')
        layer_config['filters'] = units
        layer_config['name'] += '_pruned'
        return Conv2D.from_config(layer_config)
    elif isinstance(layer, MaxPooling2D):
        layer_config['name'] += '_pruned'
        return MaxPooling2D.from_config(layer_config)
    elif isinstance(layer, AveragePooling2D):
        layer_config['name'] += '_pruned'
        return AveragePooling2D.from_config(layer_config)
    elif isinstance(layer, Flatten):
        layer_config['name'] += '_pruned'
        return Flatten.from_config(layer_config)
    elif isinstance(layer, Dropout):
        layer_config['name'] += '_pruned'
        return Dropout.from_config(layer_config)
    elif isinstance(layer, BatchNormalization):
        layer_config['name'] += '_pruned'
        return BatchNormalization.from_config(layer_config)
    else:
        raise ValueError('Unsupported layer type')


def get_input_shape(layer):
    try:
        return layer.input_shape
    except AttributeError:
        return layer._build_shapes_dict['input_shape'] # noqa


class Slicer:
    def __init__(self,
                 model_filename,
                 expanded_contrib_matrix,
                 top_percentage,
                 train_data,
                 validation_split,
                 optimizer,
                 epochs,
                 patience):
        self._model_filename = model_filename
        self._expanded_contrib_matrix = expanded_contrib_matrix
        if top_percentage <= 0:
            raise ValueError('At least 1 neuron/neuron group from each layer has to be selected')
        self._top_per = top_percentage
        self._retrain = False
        if train_data is not None:
            self._train_inputs, self._train_outputs = train_data
            if validation_split <= 0 or validation_split > 0.5:
                raise ValueError('Invalid validation split: a in range (0, 0.5] is expected')
            self._val_split = validation_split
            self._retrain = True
        self._optimizer = optimizer
        self._epochs = epochs
        self._patience = patience
        self._selected_units = set()

    def get_slice(self, output_index):  # the contrib_matrix implicitly contains data part of slice criterion info
        pass

    def get_selected_units(self):
        return self._selected_units


class OptimizedRankedSlicerCls(Slicer):
    def __init__(self,
                 model_filename,
                 expanded_contrib_matrix,
                 top_percentage,
                 train_data=None,
                 validation_split=0.1,
                 optimizer='adam',
                 epochs=64,
                 patience=-1):
        super().__init__(model_filename,
                         expanded_contrib_matrix,
                         top_percentage,
                         train_data,
                         validation_split,
                         optimizer,
                         epochs,
                         patience)

    def get_slice(self, output_index):
        expanded_contrib_matrix = self._expanded_contrib_matrix[output_index]
        original_model = load_model(self._model_filename)
        sliced_model = Sequential()
        for layer_index in range(len(original_model.layers) - 1):
            layer = original_model.layers[layer_index]
            layer.trainable = False
            n = 0
            if layer_index in expanded_contrib_matrix:
                pairs_list = expanded_contrib_matrix[layer_index]
                u = int(np.ceil(self._top_per * len(pairs_list)))
                n = len({ui for p in pairs_list[:u] for ui in p[0]})
            sliced_model.add(create_like_pruned_layer(layer, units=n))
        sliced_model.add(Dense(2,
                               activation='softmax',
                               kernel_initializer=None,
                               bias_initializer=None,
                               name='last_layer'))
        sliced_model.compile(loss='categorical_crossentropy', optimizer=self._optimizer, metrics=['acc'])
        sliced_model.build(input_shape=get_input_shape(original_model.layers[0]))
        prev_layer_index = -1
        prev_top_n = None
        prev_layer = None
        for layer_index in range(len(original_model.layers) - 1):
            if layer_index in expanded_contrib_matrix:
                layer = original_model.layers[layer_index]
                slice_layer = sliced_model.layers[layer_index]
                pairs_list = expanded_contrib_matrix[layer_index]
                n = int(np.ceil(self._top_per * len(pairs_list)))
                top_n = {ui for p in pairs_list[:n] for ui in p[0]}  # flatten selected indices
                for ui in top_n:
                    self._selected_units.add((layer_index, ui))
                layer_weights = layer.get_weights()
                units = layer_weights[0].shape[-1]
                slice_layer_weights = slice_layer.get_weights()
                i = 0
                for unit_index in range(units):
                    if unit_index in top_n:
                        if prev_top_n is not None:  # prev_layer would also be non-None
                            if isinstance(layer, Dense) and isinstance(prev_layer, Conv2D):
                                kernel_sz = int(np.prod(get_input_shape(original_model.layers[layer_index - 1])[1:-1]))
                                mask_sz = get_input_shape(original_model.layers[layer_index])[1]
                                mask = np.zeros(mask_sz, dtype=bool)
                                for x in prev_top_n:
                                    for j in range(kernel_sz):
                                        mask[x * kernel_sz + j] = True
                            else:
                                mask_sz = prev_layer.get_weights()[0].shape[-1] # noqa
                                mask = np.zeros(mask_sz, dtype=bool)
                                mask[prev_top_n] = True
                            slice_layer_weights[0][..., i] = layer_weights[0][..., unit_index][..., mask]
                            slice_layer_weights[1][i] = layer_weights[1][unit_index]
                        else:
                            slice_layer_weights[0][..., i] = layer_weights[0][..., unit_index]
                            slice_layer_weights[1][i] = layer_weights[1][unit_index]
                        i += 1
                slice_layer.set_weights(slice_layer_weights)
                prev_layer = layer
                prev_top_n = list(top_n)
                prev_layer_index = layer_index
        original_last_layer = original_model.layers[-1]
        layer_weights = original_last_layer.get_weights()
        units = layer_weights[0].shape[-1]
        sliced_last_layer = sliced_model.layers[-1]
        w = sliced_last_layer.get_weights()
        k = 0
        for unit_index in range(units):
            if prev_top_n is None:
                if unit_index == output_index:
                    w[0][:, 0] = layer_weights[0][:, unit_index]
                    w[1][0] = layer_weights[1][unit_index]
                else:
                    w[0][:, 1] += layer_weights[0][:, unit_index]
                    w[1][1] += layer_weights[1][unit_index]
                    k += 1
            else:
                mask = np.zeros(original_model.layers[prev_layer_index].units, dtype=bool)
                mask[prev_top_n] = True
                if unit_index == output_index:
                    w[0][:, 0] = layer_weights[0][:, unit_index][mask]
                    w[1][0] = layer_weights[1][unit_index]
                else:
                    w[0][:, 1] += layer_weights[0][:, unit_index][mask]
                    w[1][1] += layer_weights[1][unit_index]
                    k += 1
        if k > 0:
            w[0][:, 1] /= k
            w[1][1] /= k
        sliced_last_layer.set_weights(w)

        if self._retrain:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_model_file_name = temp_file.name + '.weights.h5'
            callbacks = [ModelCheckpoint(temp_model_file_name,
                                         monitor='val_acc',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='auto',
                                         verbose=0)]
            if self._patience > 0:
                callbacks.append(EarlyStopping(monitor='val_acc',
                                               patience=self._patience,
                                               mode='auto',
                                               verbose=0,
                                               restore_best_weights=True))
            print('Adjusting last layer weights for module #%d...' % output_index)
            sliced_model.fit(self._train_inputs,
                             self._train_outputs,
                             epochs=self._epochs,
                             validation_split=self._val_split,
                             callbacks=callbacks)
            sliced_model.load_weights(temp_model_file_name)
            os.remove(temp_model_file_name)
        return sliced_model


class OptimizedRankedSlicerReg(Slicer):
    def __init__(self,
                 model_filename,
                 expanded_contrib_matrix,
                 top_percentage,
                 train_data=None,
                 validation_split=0.1,
                 optimizer='adam',
                 epochs=64,
                 patience=-1):
        super().__init__(model_filename,
                         expanded_contrib_matrix,
                         top_percentage,
                         train_data,
                         validation_split,
                         optimizer,
                         epochs,
                         patience)

    def get_slice(self, output_index):
        expanded_contrib_matrix = self._expanded_contrib_matrix[output_index]
        original_model = load_model(self._model_filename)
        sliced_model = Sequential()
        for layer_index in range(len(original_model.layers) - 1):
            layer = original_model.layers[layer_index]
            layer.trainable = False
            n = 0
            if layer_index in expanded_contrib_matrix:
                pairs_list = expanded_contrib_matrix[layer_index]
                u = int(np.ceil(self._top_per * len(pairs_list)))
                n = len({ui for p in pairs_list[:u] for ui in p[0]})
            sliced_model.add(create_like_pruned_layer(layer, units=n))
        sliced_model.add(create_like_pruned_layer(original_model.layers[-1], units=1))
        sliced_model.compile(loss='mse', optimizer=self._optimizer, metrics=['mae'])
        sliced_model.build(input_shape=get_input_shape(original_model.layers[0]))
        prev_layer_index = -1
        prev_top_n = None
        prev_layer = None
        for layer_index in range(len(original_model.layers) - 1):
            if layer_index in expanded_contrib_matrix:
                layer = original_model.layers[layer_index]
                slice_layer = sliced_model.layers[layer_index]
                pairs_list = expanded_contrib_matrix[layer_index]
                n = int(np.ceil(self._top_per * len(pairs_list)))
                top_n = {ui for p in pairs_list[:n] for ui in p[0]}  # flatten selected indices
                for ui in top_n:
                    self._selected_units.add((layer_index, ui))
                layer_weights = layer.get_weights()
                units = layer_weights[0].shape[-1]
                slice_layer_weights = slice_layer.get_weights()
                i = 0
                for unit_index in range(units):
                    if unit_index in top_n:
                        if prev_top_n is not None:  # prev_layer would also be non-None
                            if isinstance(layer, Dense) and isinstance(prev_layer, Conv2D):
                                kernel_sz = int(np.prod(get_input_shape(original_model.layers[layer_index - 1])[1:-1]))
                                mask_sz = get_input_shape(original_model.layers[layer_index])[1]
                                mask = np.zeros(mask_sz, dtype=bool)
                                for x in prev_top_n:
                                    for j in range(kernel_sz):
                                        mask[x * kernel_sz + j] = True
                            else:
                                mask_sz = prev_layer.get_weights()[0].shape[-1] # noqa
                                mask = np.zeros(mask_sz, dtype=bool)
                                mask[prev_top_n] = True
                            slice_layer_weights[0][..., i] = layer_weights[0][..., unit_index][..., mask]
                            slice_layer_weights[1][i] = layer_weights[1][unit_index]
                        else:
                            slice_layer_weights[0][..., i] = layer_weights[0][..., unit_index]
                            slice_layer_weights[1][i] = layer_weights[1][unit_index]
                        i += 1
                slice_layer.set_weights(slice_layer_weights)
                prev_layer = layer
                prev_top_n = list(top_n)
                prev_layer_index = layer_index
        original_last_layer = original_model.layers[-1]
        layer_weights = original_last_layer.get_weights()
        units = layer_weights[0].shape[-1]
        sliced_last_layer = sliced_model.layers[-1]
        w = sliced_last_layer.get_weights()
        for unit_index in range(units):
            if prev_top_n is None:
                if unit_index == output_index:
                    w[0][:, 0] = layer_weights[0][:, unit_index]
                    w[1][0] = layer_weights[1][unit_index]
            else:
                mask = np.zeros(original_model.layers[prev_layer_index].units, dtype=bool)
                mask[prev_top_n] = True
                if unit_index == output_index:
                    w[0][:, 0] = layer_weights[0][:, unit_index][mask]
                    w[1][0] = layer_weights[1][unit_index]
        sliced_last_layer.set_weights(w)
        if self._retrain:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_model_file_name = temp_file.name + '.weights.h5'
            callbacks = [ModelCheckpoint(temp_model_file_name,
                                         monitor='val_mae',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='auto',
                                         verbose=0)]
            if self._patience > 0:
                callbacks.append(EarlyStopping(monitor='val_mae',
                                               patience=self._patience,
                                               mode='auto',
                                               verbose=0,
                                               restore_best_weights=True))
            print('Adjusting last layer weights for module #%d...' % output_index)
            sliced_model.fit(self._train_inputs,
                             self._train_outputs,
                             epochs=self._epochs,
                             validation_split=self._val_split,
                             callbacks=callbacks)
            sliced_model.load_weights(temp_model_file_name)
            os.remove(temp_model_file_name)
        return sliced_model
