class Mutation:
    def __init__(self, uid, location, mutant, description):
        self._uid = uid
        self._location = location
        self._mutant = mutant
        self._description = description

    def get_uid(self):
        return self._uid

    def get_location(self):
        return self._location

    def get_mutant(self):
        return self._mutant

    def get_description(self):
        return self._description


class Location:
    def __init__(self, layer_index, unit_indices):
        self._layer_index = layer_index
        self._unit_indices = unit_indices

    def get_layer_index(self):
        return self._layer_index

    def get_unit_indices(self):
        return self._unit_indices
