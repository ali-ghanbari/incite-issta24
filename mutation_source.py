from keras.models import load_model
from tqdm import tqdm

from mutation import Mutation, Location
from util import model_ok, MutationUIDGenerator


class MutationSource:
    def get_mutations(self, consumers):
        pass


class MutationGenerator(MutationSource):
    def __init__(self, model_filename, mutable_units, fraction):
        model = load_model(model_filename)
        if not model_ok(model):
            raise ValueError('Unsupported model type')
        self._mutable_units = mutable_units
        self._consumers = []
        self._fraction = fraction
        self._uid_gen = MutationUIDGenerator()

    def get_mutations(self, consumers):
        if not consumers:
            self._consumers = []
        else:
            self._consumers = consumers
        self._apply_mutators()

    def _submit_mutation(self, uid, location, model, description):
        mutation = Mutation(uid, location, model, description)
        for consumer in self._consumers:
            consumer.consume(mutation)

    def _apply_mutators(self):
        progress_bar = tqdm(total=3 * len(self._mutable_units), position=0, leave=True, desc="Analyzing Mutants")
        for mu in self._mutable_units:
            location = Location(mu.get_layer_index(), mu.get_unit_indices())
            mu.add(self._fraction)
            self._submit_mutation(str(self._uid_gen.generate()),
                                  location,
                                  mu.get_model(),
                                  'Incremented weights by %.2f%%' % (self._fraction * 100))
            mu.reset()
            progress_bar.update(1)
            mu.add(-self._fraction)
            self._submit_mutation(str(self._uid_gen.generate()),
                                  location,
                                  mu.get_model(),
                                  'Decremented weights by %.2f%%' % (self._fraction * 100))
            mu.reset()
            progress_bar.update(1)
            mu.add(-1.0)
            self._submit_mutation(str(self._uid_gen.generate()),
                                  location,
                                  mu.get_model(),
                                  'Deleted weights')
            mu.reset()
            progress_bar.update(1)
