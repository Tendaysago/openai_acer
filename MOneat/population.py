"""Implements the core evolution algorithm."""
from __future__ import print_function

from MOneat.reporting import ReporterSet
from MOneat.math_util import mean
from MOneat.six_util import iteritems, itervalues
import datetime
import os
import math
import random
import numpy


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        self.title = config.reproduction_config.multi_optimization
        if config.reproduction_config.multi_optimization_indicator is not None:
            self.title += config.reproduction_config.multi_optimization_indicator
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(
            config.reproduction_config, self.reporters, stagnation
        )

        self.outpath = "{}_{}".format(
            self.title,
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        )
        self.multi_dir = self.reproduction.reproduction_config.multi_optimization
        try:
            os.makedirs(self.outpath)
        except FileExistsError:
            pass
        if config.fitness_criterion == "max":
            self.fitness_criterion = max
        elif config.fitness_criterion == "min":
            self.fitness_criterion = min
        elif config.fitness_criterion == "mean":
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion)
            )

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(
                config.genome_type, config.genome_config, config.pop_size
            )
            self.species = config.species_set_type(
                config.species_set_config, self.reporters
            )
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
            self.allhistory = []
        else:
            (
                self.population,
                self.species,
                self.generation,
                self.allhistory,
            ) = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination"
            )

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation, self.outpath)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            if self.reproduction.reproduction_config.multi_optimization == "NEATPS":
                caluculatePS(list(itervalues(self.population)), self.multi_dir)

            self.reporters.post_evaluate(
                self.config, self.population, self.species, best
            )

            # Track the best genome ever seen.
            if self.reproduction.reproduction_config.multi_optimization == "None":
                if self.best_genome is None or best.fitness > self.best_genome.fitness:
                    self.best_genome = best
            else:
                self.best_genome = self.allhistory

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(
                    g.fitness for g in itervalues(self.population)
                )
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population, self.allhistory = self.reproduction.reproduce(
                self.config,
                self.species,
                self.config.pop_size,
                self.generation,
                self.allhistory,
            )
            # print(self.allhistory)
            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type,
                        self.config.genome_config,
                        self.config.pop_size,
                    )
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(
                self.config,
                self.population,
                self.species,
                self.allhistory,
                self.outpath,
            )

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome
            )

        return self.best_genome


def caluculatePS(individuals, multi_dir):
    """Apply SPEA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *n* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *n* will have no effect other
    than sorting the population according to a strength Pareto scheme. The
    list returned contains references to the input *individuals*. For more
    details on the SPEA-II operator see [Zitzler2001]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    .. [Zitzler2001] Zitzler, Laumanns and Thiele, "SPEA 2: Improving the
    strength Pareto evolutionary algorithm", 2001.
    """
    N = len(individuals)
    strength_fit = [0] * N
    fits = [0] * N
    dominating_inds = [list() for i in range(N)]
    inds = []
    for ind in individuals:
        inds.append(ind.fitness)
    if multi_dir == "multi_min":
        wobj = numpy.array([(ind[1].fitness) for ind in individuals])
        wobj = numpy.array(1.0 - wobj)
        inds = wobj.tolist()
    for i, ind_i in enumerate(inds):
        for j, ind_j in enumerate(inds[i + 1 :], i + 1):
            if isdominates(ind_i, ind_j):
                strength_fit[i] += 1
                dominating_inds[j].append(i)
            elif isdominates(ind_j, ind_i):
                strength_fit[j] += 1
                dominating_inds[i].append(j)

    for i in range(N):
        for j in dominating_inds[i]:
            fits[i] += strength_fit[j]
        individuals[i].pareto_strength = fits[i]


def _randomizedSelect(array, begin, end, i):
    """Allows to select the ith smallest element from array without sortingit.
    Runtime is expected to be O(n).
    """
    if begin == end:
        return array[begin]

    q = _randomizedPartition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomizedSelect(array, begin, q, i)
    else:
        return _randomizedSelect(array, q + 1, end, i - k)


def _randomizedPartition(array, begin, end):
    i = random.randint(begin, end)
    array[begin], array[i] = array[i], array[begin]
    return _partition(array, begin, end)


def _partition(array, begin, end):
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


def isdominates(fitnesses1, fitnesses2):
    """Return true if each objective of *self* is not strictly worse than
    the corresponding objective of *other* and at least one objective is
    strictly better.

    :param obj: Slice indicating on which objectives the domination is
                tested. The default value is `slice(None)`, representing
                every objectives.
    """
    not_equal = False
    for self_fitnesses, other_fitnesses in zip(fitnesses1, fitnesses2):
        if self_fitnesses > other_fitnesses:
            not_equal = True
        elif self_fitnesses < other_fitnesses:
            return False
    return not_equal
