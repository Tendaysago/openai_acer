"""Keeps track of whether species are making progress and helps remove ones that are not."""
import sys

from MOneat.config import ConfigParameter, DefaultClassConfig
from MOneat.six_util import iteritems
from MOneat.math_util import stat_functions
import numpy
# TODO: Add a method for the user to change the "is stagnant" computation.

class DefaultStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('species_fitness_func', str, 'mean'),
                                   ConfigParameter('species_priority_fitness_func', str,'mean'),
                                   ConfigParameter('max_stagnation', int, 15),
                                   ConfigParameter('species_elitism', int, 0)])

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        self.species_priority_fitness_func = stat_functions.get(config.species_priority_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(config.species_fitness_func))

        self.reporters = reporters

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """
        species_data = []
        prev_ref_points = None
        for sid, s in iteritems(species_set.species):
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            if s.priority_fitness_history:
                prev_priority_fitness = max(s.priority_fitness_history)  
            else:
                if self.species_fitness_func is stat_functions['momean']:
                    prev_fitness = []
                    for _ in range(len(s.get_fitnesses()[0])):
                        prev_fitness.append(-sys.float_info.max)
                else:
                    prev_fitness = -sys.float_info.max
                if self.species_priority_fitness_func is stat_functions['momean']:
                    prev_priority_fitness = []
                    for _ in range(len(s.get_priority_fitnesses()[0])):
                        prev_priority_fitness.append(-sys.float_info.max)
                else:
                    prev_priority_fitness = -sys.float_info.max
            s.fitness = self.species_fitness_func(s.get_fitnesses(),len(s.get_fitnesses()[0]))
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            s.priority_fitness = self.species_priority_fitness_func(s.get_priority_fitnesses(),len(s.get_fitnesses()[0]))
            s.priority_fitness_history.append(s.priority_fitness)
            if s.ref_points is None:
                s.ref_points = numpy.full(len(s.get_fitnesses()[0]),1.0)
            if s.best_fitness is None:
                s.best_fitness = s.fitness
            if s.best_ref_points is None:
                s.best_ref_points = s.ref_points
            if s.single_best_fitnesses is None:
                s.single_best_fitnesses = s.fitness
            #if prev_fitness is None or s.fitness > prev_fitness:
            if isdominated(s.best_fitness,s.fitness):
                s.last_improved = generation
                s.best_fitness = s.fitness
            #if prev_priority_fitness is None or s.priority_fitness > prev_priority_fitness:
            #    s.last_improved = generation
            if isdominated(s.ref_points, s.best_ref_points):
                s.last_improved = generation
                s.best_ref_points = s.ref_points
            for d in range(len(s.single_best_fitnesses)):
                if(s.fitness[d]>s.single_best_fitnesses[d]):
                    s.single_best_fitnesses[d] = s.fitness[d]
                    s.last_improved = generation
            if s.ref_points is not None:
                prev_ref_points = s.ref_points
            
            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation

            if (len(species_data) - idx) <= self.stagnation_config.species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result

def isdominated(fitnesses1, fitnesses2):
    """Returns whether or not *wvalues1* dominates *wvalues2*.

    :param wvalues1: The weighted fitness values that would be dominated.
    :param wvalues2: The weighted fitness values of the dominant.
    :returns: :obj:`True` if wvalues2 dominates wvalues1, :obj:`False`
              otherwise.
    """

    not_equal = False
    for self_fitnesses, other_fitnesses in zip(fitnesses1,fitnesses2):
        if self_fitnesses > other_fitnesses:
            return False
        elif self_fitnesses < other_fitnesses:
            not_equal = True
    return not_equal