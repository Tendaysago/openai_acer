"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division

import math
import numpy
import random
import bisect
from itertools import count,chain
from collections import defaultdict, namedtuple
from operator import attrgetter, itemgetter
from MOneat.config import ConfigParameter, DefaultClassConfig
from MOneat.math_util import mean
from MOneat.six_util import iteritems, itervalues

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.

class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2),
                                   ConfigParameter('multi_optimization', str, 'None')])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    """
    def reproduce(self, config, species, pop_size, generation):
        #Handles creation of genomes, either from scratch or by sexual or
        #asexual reproduction from parents.
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {} # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size,self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population

    """

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {} # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size,self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)

            if self.reproduction_config.multi_optimization=='None' or repro_cutoff<self.reproduction_config.NSGA2threshold:
                # Sort members in order of descending fitness.
                old_members.sort(reverse=True, key=lambda x: x[1].fitness)

                # Transfer elites to new generation.
                if self.reproduction_config.elitism > 0:
                    for i, m in old_members[:self.reproduction_config.elitism]:
                        new_population[i] = m
                        spawn -= 1

                if spawn <= 0:
                    continue

                old_members = old_members[:repro_cutoff]

                # Randomly choose parents and produce the number of offspring allotted to the species.
            
            elif self.reproduction_config.multi_optimization=='NSGA2' and repro_cutoff>=self.reproduction_config.NSGA2threshold:

                old_members = selNSGA2(old_members, repro_cutoff, self.reproduction_config.NSGA2nd)

            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population

    

def selNSGA2(individuals, k, nd='standard'):
    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k- len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen

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
            not_equal = False
    return not_equal

def sortNondominated(individuals, k, first_front_only=False,reverse=False):
    if k==0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = map_fit_ind.keys()

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if map_fit_ind[fit_i].dominates(map_fit_ind[fit_j]):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif map_fit_ind[fit_j].dominates(map_fit_ind[fit_i]):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -=1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def sortLogNondominated(individuals, k, first_front_only=False):
    """Sort *individuals* in pareto non-dominated fronts using the Generalized
    Reduced Run-Time Complexity Non-Dominated Sorting Algorithm presented by
    Fortin et al. (2013).

    :param individuals: A list of individuals to select from.
    :returns: A list of Pareto fronts (lists), with the first list being the
              true Pareto front.
    """
    if k==0:
        return []
    
    #Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    for i, ind in enumerate(individuals):
        unique_fits[ind.fitness].append(ind)
    
    #Launch the sorting algorithm
    obj = len(individuals[0].fitness)-1
    fitnesses = unique_fits.keys()
    front = dict.fromkeys(fitnesses,0)

    #Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA(fitnesses,obj, front)

    #Extract individuals from front list here
    nbfronts = max(front.values())+1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])
    
    # Keep only the fronts required to have k individuals.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:
                return pareto_fronts[:i+1]
        return pareto_fronts
    else:
        return pareto_fronts[0]

def sortNDHelperA(fitnesses, obj, front):
    """Create a non-dominated sorting of S on the first M objectives"""
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        #Only two individulas, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if isdominated(s2[:obj+1],s1[:obj+1]):
            front[s2] = max(front[s2],front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        #All individuals for objective M are equal: go to objective M-1
        sortNDHelperA(fitnesses, obj-1, front)
    else:
        # More than two individuals for objective M are equal: go to objective M-1
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA(best,obj, front)
        sortNDHelperB(best,worst, obj-1, front)
        sortNDHelperA(worst, obj, front)

def sortNDHelperB(best, worst, obj, front):
    """Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called."""
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) ==0:
        #One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        #One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if isdominated(hi[:obj+1], li[:obj+1]) or hi[:obj+1] == li[:obj+1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst,key=key)):
        #All individuals from L dominate H for objective M:
        #Also supports the case where every individuals in L and H
        #has the same value for the current objective
        #Skip to objective M-1
        sortNDHelperB(best,worst, obj-1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best,worst,obj)
        sortNDHelperB(best1, worst1, obj, front)
        sortNDHelperB(best1, worst2, obj-1, front)
        sortNDHelperB(best2, worst2, obj, front)

def splitA(fitnesses, obj):
    """Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = [], []
    best_b, worst_b = [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)
    
    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b

def splitB(best, worst, obj):
    """Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)
    
    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)
    
    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b

def sweepA(fitnesses, front):
    """Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair]+1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx,fit)

def sweepB(best, worst, front):
    """Adjust the rank number of the worst fitnesses according to
    the best fitnesses on the first two objectives using a sweep
    procedure.
    """
    stairs, fstairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, False)
        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair]+1)
 
def assignCrowdingDist(individuals):


    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness, i ) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm
    
    for i, dist in enumerate(distances):
        individuals[i].crowding_dist = dist

def selTournamentDCD(self,individuals, k):
    """Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4. Starting from the beginning of the selected
    individuals, two consecutive individuals will be different (assuming all
    individuals in the input list are unique). Each individual from the input
    list won't be selected more than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """

    if len(individuals) % 4 != 0:
        raise ValueError("selTournamentDCD: individuals length must be a multiple of 4")

    if k % 4 != 0:
        raise ValueError("selTournamentDCD: number of individuals to select must be a multiple of 4")
    
    def tourn(ind1, ind2):
        if ind1.dominates(ind2):
            return ind1
        elif ind2.dominates(ind1):
            return ind2

        if ind1.crowding_dist < ind2.crowding_dist:
            return ind2
        elif ind1.crowding_dist > ind2.crowding_dist:
            return ind1
        
        if random.random() <= 0.5:
            return ind1
        return ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k , 4):
        chosen.append(tourn(individuals_1[i],   individuals_1[i+1]))
        chosen.append(tourn(individuals_1[i+2], individuals_1[i+3]))
        chosen.append(tourn(individuals_2[i],   individuals_2[i+1]))
        chosen.append(tourn(individuals_2[i+2], individuals_2[i+3]))

    return chosen

def selSPEA2(individuals, k):
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
    L = len(individuals[0].fitness)
    K = math.sqrt(N)
    strength_fit = [0] * N
    fits = [0] * N
    dominating_inds = [list() for i in range(N)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i+1:], i+1):
            if ind_i.dominates(ind_j):
                strength_fit[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.dominates(ind_i):
                strength_fit[j] += 1
                dominating_inds[i].append(j)

    
    for i in range(N):
        for j in dominating_inds[i]:
            fits[i] += strength_fit[j]

    chosen_indices = [i for i in range(N) if fits[i] < 1]

    if len(chosen_indices) < k:
        for i in range(N):
            distances = [0.0] * N
            for j in range(i+1, N):
                dist = 0.0
                for l in range(L):
                    val = individuals[i].fitness[l] - individuals[j].fitness[l]
                    dist += val * val
                distances[j] = dist
            kth_dist = _randomizedSelect(distances, 0, N-1, K)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density

        next_indices = [(fits[i], i) for i in range(N) if not i in chosen_indices]
        next_indices.sort()

        chosen_indices += [i for _, i in next_indices[:k - len(chosen_indices)]]

    elif len(chosen_indices) > k: # The archive is too large
        N = len(chosen_indices)
        distances = [[0.0] * N for i in range(N)]
        sorted_indices = [[0] * N for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                dist = 0.0
                for l in range(L):
                    val = individuals[chosen_indices[i]].fitness[l] - individuals[chosen_indices[j]].fitness[l]
                    dist += val * val
                distances[i][j] = dist
                distances[j][i] = dist
            distances[i][i] = -1

        # Insert sort is faster than quick sort for short arrays
        for i in range(N):
            for j in range(1, N):
                l = j
                while l > 0 and distances[i][j] < distances[i][sorted_indices[i][l-1]]:
                    sorted_indices[i][l] = sorted_indices[i][l-1]
                    l -= 1
                sorted_indices[i][l] = j

        size = N
        to_remove = []
        while size > k:
            # Search for minimal distance
            min_pos = 0
            for i in range(1,N):
                for j in range(1, size):
                    dist_i_sorted_j = distances[i][sorted_indices[i][j]]
                    dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

                    if dist_i_sorted_j < dist_min_sorted_j:
                        min_pos = i
                        break
                    elif dist_i_sorted_j > dist_min_sorted_j:
                        break

            # Remove minimal distance from sorted_indices
            to_remove.append(min_pos)
            size -= 1

        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]

    return [individuals[i] for i in chosen_indices]

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
        return _randomizedSelect(array, q+1, end, i-k)

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

NSGA3Memory = namedtuple("NSGA3Memory", ["best_point", "worst_point", "extreme_points"])

class selNSGA3WithMemory(object):
    """Class version of NSGA-III selection including memory for best, worst and
    extreme points. 
    """

    def __init__(self, ref_points, nd="log"):
        self.ref_points = ref_points
        self.nd = nd
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points = None

    def __call__(self, individuals, k):
        chosen, memory = selNSGA3(invididuals, k, self.ref_points, self.nd, self.best_point, 
                                self.worst_point, self.extreme_points, True)
        self.best_point = memory.best_point.reshape((1,-1))
        self.worst_point = memory.worst_point.reshape((1,-1))
        self.extreme_points = memory.extreme_points
        return chosen

def selNSGA3(individuals, k, ref_points, nd="log", best_point=None,
            worst_point=None, extreme_points=None, return_memory=False):
    """Implementation of NSGA-III selection as presented in [Deb2014]_.

    This implementation is partly based on `lmarti/nsgaiii
    <https://github.com/lmarti/nsgaiii>`_. It departs slightly from the
    original implementation in that it does not use memory to keep track
    of ideal and extreme points. 

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param ref_points: Reference points to use for niching.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :param best_point: Best point found at previous generation. If not provided
        find the best point only from current individuals.
    :param worst_point: Worst point found at previous generation. If not provided
        find the worst point only from current individuals.
    :param extreme_points: Extreme points found at previous generation. If not provided
        find the extreme points only from current individuals.
    :param return_memory: If :data:`True`, return the best, worst and extreme points
        in addition to the chosen individuals.
    :returns: A list of selected individuals.
    :returns: If `return_memory` is :data:`True`, a namedtuple with the
        `best_point`, `worst_point`, and `extreme_points`.

    You can generate the reference points using the :func:`uniform_reference_points`
    function::

        >>> ref_points = tools.uniform_reference_points(nobj=3, p=12)   # doctest: +SKIP
        >>> selected = selNSGA3(population, k, ref_points)              # doctest: +SKIP

    .. [Deb2014] Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
        Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
        Part I: Solving Problems With Box Constraints. IEEE Transactions on
        Evolutionary Computation, 18(4), 577-601. doi:10.1109/TEVC.2013.2281535.
    """    
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception("selfNSGA3: The choice of non-dominated sorting "
                        "method '{0}' is invalid.".format(nd))
    # Extract fitnesses as a numpy array in the nd-sort order
    # Use wvalues * -1 to tackle always as a minimization problem
    fitnesses = numpy.array([ind.fitness for f in pareto_fronts for ind in f])
    fitnesses *= -1

    # Get best and worst point of population, contrary to pymoo
    # we don't use memory
    if best_point is not None and worst_point is not None:
        best_point  = numpy.min(numpy.concatenate((fitnesses, best_point), axis=0), axis=0)
        worst_point = numpy.max(numpy.concatenate((fitnesses, worst_point),axis=0), axis=0)
    else:
        best_point  = numpy.min(fitnesses, axis=0)
        worst_point = numpy.max(fitnesses, axis=0)
    
    extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
    front_worst = numpy.max(fitnesses[:sum(len(f) for f in pareto_fronts), :] axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    # Get counts per niche for individuals in all front but the last
    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    index, counts = numpy.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    # Choose individuals from all fronts but the last
    chosen = list(chain(*pareto_fronts[:-1]))

    # Use niching to select the remaining individuals
    sel_count = len(chosen)
    n = k- sel_count
    selected = niching(pareto_fronts[-1], n, niches[self_count:], dist[sel_count:], niche_counts)
    chosen.extend(selected)

    if return_memory:
        return chosen, NSGA3Memory(best_point, worst_point, extreme_points)
    return chosen

def find_extreme_points(fitnesses, best_point, extreme_points=None):
    'Finds the individuals with extreme values for each objective function.'
    # Keep track of last generation extreme points
    if extreme_points is not None:
        fitnesses = numpy.concatenate((fitnesses, extreme_points), axis=0)
    
    # Translate objectives
    ft = fitnesses - best_point

    # Find achievement scalarizing funtion(asf)
    asf = numpy.eye(best_point.shape[0])
    asf[asf==0] = 1e6
    asf = numpy.max(ft* asf[:, numpy.newaxis, :], axis=2)

    # Extreme point are the fitnesses with minimal asf
    min_asf_idx = numpy.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]

def find_intercepts(extreme_points, best_point, current_worst, front_worst):
    """Find intercepts between the hyperplane and each axis with 
    the ideal point as origin."""
    # Construct hyperplane sum(f_i^n) = 1
    b = numpy.ones(extreme_points.shape[1])
    A = extreme_points- best_point
    try:
        x = numpy.linalg.solve(A, b)
    except numpy.linalg.LinAlgError:
        intercepts = current_worst
    else:
        intercepts = 1 / x

        if(not numpy.allclose(numpy.dot(A, x), b) or
                numpy.any(intercepts <= 1e-6) or
                numpy.any((intercepts + best_point) > current_worst)):
            intercepts = front_worst

    return intercepts

def associate_to_niche(fitnesses, reference_points, best_point, intercepts):
    """Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jaiun(2014)."""
    # Normalize by ideal point and intercepts
    fn = (fitnesses - best_point) / (intercepts - best_point)

    #Create distance matrix
    fn = numpy.repeat(numpy.expand_dims(fn, axis=1), len(reference_points), axis=1)
    distances = numpy.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    distances = distances[:, :, numpy.newaxis] * reference_points[numpy.newaxis, :, :] / norm[numpy.newaxis, :, numpy.newaxis]
    distances = numpy.linalg.norm(distances - fn, axis=2)

    # Retrieve min distance niche index
    niches = numpy.argmin(distances, axis=1)
    distances = distances[range(niches.shape[0]), niches]
    return niches, distances

def niching(individuals, k, niches, distances, niche_counts):
    selected = []
    available = numpy.ones(len(individuals), dtype = numpy.bool)
    while len(selected) < k:
        # Maximum number of individuals (niches) to select in that round
        n = k- len(selected)

        # Find the available niches and the minimum niche count in them
        available_niches = numpy.zeros(len(niche_counts), dtype=numpy.bool)
        available_niches[numpy.unique(nichs[available])] = True
        min_count = numpy.min(niche_counts[available_niches])

        # Select at most n niches with the minimum count
        selected_niches = numpy.flatnonzero(numpy.logical_and(available_niches, niche_counts == min_count))
        numpy.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            # Find the individuals associated with this niche
            niche_individuals = numpy.flatnonzero(niches==niche)
            numpy.random.shuffle(niche_individuals)

            # If no individual in that niche, select the closest to reference
            # Else select randomly
            if niche_counts[niche] ==0:
                sel_index = niche_individuals[numpy.argmin(distances[niche_individuals])]
            else:
                sel_index = niche_individuals[0]

            # Update availability, counts and selection
            available[sel_index] = False
            niche_counts[niche]+=1
            selected.append(individuals[sel_index])

    return selected

def uniform_reference_points(nobj, p=4, scaling=None):
    """Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj -1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i/ total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left-i. total. depth+1))
        return points

    ref_points = numpy.array(gen_refs_recursive(numpy.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *~ scaling
        ref_points += (1- scaling) / nobj

    return ref_points