import numpy as np
from keras.layers import Dense, Conv2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from util import model_ok, make_axes


class UnitClustering:
    def __init__(self, model):
        if not model_ok(model):
            raise ValueError('An instance of Sequential or Model is expected')
        # if not is_softmax_classifier(model):
        #     raise ValueError('A classifier with softmax output is expected')
        self._model = model

    def get_clusters(self, cluster_sz):
        if cluster_sz <= 0:
            raise ValueError('Cluster size must be a positive number')
        clusters = []
        for layer_index in range(len(self._model.layers) - 1):  # exclude last layer
            layer = self._model.layers[layer_index]

            if not isinstance(layer, Dense) and not isinstance(layer, Conv2D):
                continue

            w = layer.get_weights()
            a = w[0]
            b = w[1]
            b_expanded = np.expand_dims(b, axis=make_axes(a))
            b_tiled = np.tile(b_expanded, reps=(*a.shape[:-2], 1, 1))
            points = np.concatenate((a, b_tiled), axis=-2)
            points = points.transpose()
            points = points.reshape((points.shape[0], -1))
            points = (MinMaxScaler()).fit_transform(points)

            num_units = w[0].shape[-1]
            num_clusters = int(np.ceil(float(num_units) / float(cluster_sz)))
            clustering = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(points)

            cluster_index_map = dict()
            for unit_index in range(len(points)):
                cluster = clustering[unit_index]  # what is the cluster index of the jth neuron
                if cluster not in cluster_index_map:
                    cluster_index_map[cluster] = []
                cluster_index_map[cluster].append(unit_index)
            for (_, unit_indices) in cluster_index_map.items():
                clusters.append(MutableUnitCluster(self._model, layer_index, unit_indices))
        return clusters


class MutableUnitCluster:
    def __init__(self, model, layer_index, unit_indices):
        self._model = model
        if len(unit_indices) == 0:
            raise ValueError('A cluster must contain at least one unit')
        self._unit_indices = unit_indices
        self._layer_index = layer_index
        self._layer = model.layers[layer_index]
        if not isinstance(self._layer, Dense) and not isinstance(self._layer, Conv2D):
            raise ValueError('Only Dense or Conv2D layers can be in a mutable cluster')
        self._original_weights_dict = dict()
        w = self._layer.get_weights()
        for i in unit_indices:
            if i in self._original_weights_dict:
                raise ValueError('Duplicate unit index')
            self._original_weights_dict[i] = (w[0][..., i] * 1.0, w[1][i])

    def add(self, fraction):
        if fraction < -1 or fraction > 1:
            raise ValueError('A fraction value in interval [-1, 1] is required')
        w = self._layer.get_weights()
        for i in self._unit_indices:
            w[0][..., i] += w[0][..., i] * fraction  # input weights
            w[1][i] += w[1][i] * fraction  # biases
        self._layer.set_weights(w)
        return self._model

    def reset(self):
        w = self._layer.get_weights()
        for i in self._unit_indices:
            w[0][..., i] = self._original_weights_dict[i][0] * 1.0  # reset weights
            w[1][i] = self._original_weights_dict[i][1]  # reset biases
        self._layer.set_weights(w)
        return self._model

    def get_model(self):
        return self._model

    def get_layer_index(self):
        return self._layer_index

    def get_unit_indices(self):
        return self._unit_indices
