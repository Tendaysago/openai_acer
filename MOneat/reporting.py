"""
Makes possible reporter classes,
which are triggered on particular events and may provide information to the user,
may do something else such as checkpointing, or may do both.
"""
from __future__ import division, print_function

import time
import datetime
from math import sqrt
from MOneat.math_util import mean, stdev, momean, mostdev
from MOneat.six_util import itervalues, iterkeys

# TODO: Add a curses-based reporter.


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config, population, species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config, population, species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config, generation, best):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.txt')

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.filewrite('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            self.filewrite('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size       fitness      priority_fitness    adj fit       stag")
            print("  ====  ===  ====  =================      =======     ===============  ====")
            self.filewrite("   ID   age  size       fitness      priority_fitness    adj fit       stag")
            self.filewrite("  ====  ===  ====  =================      =======     ===============  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                #f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                f = None
                af = None
                if s.fitness:
                    f = [round(s.fitness[i], 4) for i in range(len(s.fitness))] 
                    f = "{0}".format(f)
                #f = "--" if s.fitness is None else "{0}".format(s.fitness)
                else:
                    f = "--"
                #af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                if s.adjusted_fitness:
                    af = [round(s.adjusted_fitness[i], 4) for i in range(len(s.adjusted_fitness))] 
                    af = "{0}".format(af)
                #af = "--" if s.adjusted_fitness is None else "{0}".format(s.adjusted_fitness)
                else:
                    af = "--"
                if s.ref_points is not None:
                    r_pts = [round(s.ref_points[i], 4) for i in range(len(s.ref_points))]
                    r_pts = "{0}".format(r_pts)
                else:
                    r_pts = "--"
                pf = "--" if s.priority_fitness is None else "{:.4f}".format(s.priority_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >4}  {: >4}  {: >4}  {: >4}  {: >4}".format(sid, a, n, f, r_pts, pf, af, st))
                self.filewrite("  {: >4}  {: >3}  {: >4}  {: >4}  {: >4}  {: >4}  {: >4}  {: >4}".format(sid, a, n, f, r_pts, pf, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))
            self.filewrite('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
            self.filewrite("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))
            self.filewrite("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        #fit_mean = mean(fitnesses)
        #fit_std = stdev(fitnesses)
        #print('Population\'s average fitness: {0} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        if(type(fitnesses[0]) is float):
            fit_mean = mean(fitnesses)
            fit_std = stdev(fitnesses)
            print('Population\'s average fitness: {0:3.3f} stdev: {1:3.3f}'.format(fit_mean, fit_std))
            self.filewrite('Population\'s average fitness: {0:3.3f} stdev: {1:3.3f}'.format(fit_mean, fit_std))
        elif(type(fitnesses[0]) is list):
            fit_mean = momean(fitnesses,len(fitnesses[0]))
            fit_std = mostdev(fitnesses,len(fitnesses[0]))
            #fit_std = sqrt(sum((v - fit_mean) ** 2 for v in fitnesses) / len(fitnesses))
            print('Population\'s average fitness: {0} stdev: {1}'.format(fit_mean, fit_std))
            self.filewrite('Population\'s average fitness: {0} stdev: {1}'.format(fit_mean, fit_std))

        best_species_id = species.get_species_id(best_genome.key)
        print(
            'Best fitness: {0} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))
        self.filewrite(
            'Best fitness: {0} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')
        self.filewrite('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))
            self.filewrite("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)

    def filewrite(self,contents):
        with open(self.filename, mode='a') as f:
            f.write(contents+'\n')
