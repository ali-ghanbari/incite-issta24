import time
import numpy as np


class MutationConsumer:
    def consume(self, mutation):
        pass


class MutationCounter(MutationConsumer):
    def __init__(self):
        self._mutants_count = 0
        self._measurements = []
        self._ts = -1
        self._summaries = []

    def reset(self):
        if len(self._measurements) > 0:
            avg_time = np.mean(self._measurements[1:]) / 1000  # average in seconds
            self._summaries.append((self._mutants_count, avg_time))
            self._mutants_count = 0
            self._measurements = []
            self._ts = -1

    def consume(self, mutation):
        ts = round(time.time() * 1000)
        if self._ts < 0:
            self._ts = ts
        self._measurements.append(ts - self._ts)
        self._mutants_count += 1

    def __str__(self):
        n = len(self._summaries)
        if n == 0:
            return 'Empty\n'
        out = ''
        total_mutants = 0
        for r in range(n):
            out = 'Round %d:\n' % (r + 1)
            sp = self._summaries[r]
            out += '\t# Mutants: %d\n' % sp[0]
            out += '\tAverage Mutation Exec. Time: %fs\n' % sp[1]
            total_mutants += sp[0]
        out += '--------\n'
        out += 'Total Mutants: %d\n' % total_mutants
        return out
