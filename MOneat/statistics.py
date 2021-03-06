"""
Gathers (via the reporting interface) and provides (to callers and/or a file)
the most-fit genomes and information on genome/species fitness and species sizes.
"""
import copy
import csv
import os
import pickle
import datetime
import pandas as pd
from MOneat.math_util import mean, stdev, median2
from MOneat.reporting import BaseReporter
from MOneat.six_util import iteritems
from MOneat.reproduction import hypervolume_totalhv
from . import visualize

timestamp = None
globaloutpath = None
# TODO: Make a version of this reporter that doesn't continually increase memory usage.
# (Maybe periodically write blocks of history to disk, or log stats in a database?)


class StatisticsReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """

    def __init__(self):
        BaseReporter.__init__(self)
        self.most_fit_genomes = []
        self.generation_statistics = []
        # self.generation_cross_validation_statistics = []
        self.species_now_all_fitness = []
        self.allpareto_history = []
        self.hvhistory = []
        self.outnetworknum_history = []
        self.genome_dist_history = []

    def end_generation(
        self, config, population, species_set, allhistory=None, outpath=None
    ):
        global globaloutpath
        globaloutpath = outpath
        self.allpareto_history = allhistory
        allhistory_hv = hypervolume_totalhv(allhistory)
        allhistory_hv = round(allhistory_hv, 4)
        self.hvhistory.append(allhistory_hv)
        self.outnetworknum_history.append(len(allhistory))
        generation = str(len(self.most_fit_genomes))
        if (
            config.output_interval > 0
            and len(self.most_fit_genomes) % config.output_interval == 0
        ):
            try:
                genoutpath = outpath + "/Gen_" + generation
                os.makedirs(genoutpath)
            except FileExistsError:
                pass
            try:
                netoutpath = genoutpath + "/Networks"
                os.makedirs(netoutpath)
            except FileExistsError:
                pass
            current_pareto_plot_data = [
                list(front.values()) for front in self.species_now_all_fitness[0]
            ]
            # print(current_pareto_plot_data)
            visualize.plot_stats3D(
                len(self.most_fit_genomes),
                current_pareto_plot_data,
                filename="{}/current_generation_paretofront_fitness.png".format(
                    genoutpath
                ),
                title="Gen" + generation + "_paretofront_species",
            )
            # print(self.allpareto_history)
            allhistory_pareto_plot_data = [
                [ind[1].fitness for ind in self.allpareto_history]
            ]
            # print(allhistory_pareto_plot_data)
            visualize.plot_stats3D(
                len(self.most_fit_genomes),
                allhistory_pareto_plot_data,
                filename="{}/allhistory_paretofront_fitness".format(genoutpath),
                title="Gen" + generation + "_allhistory_pareto_front",
            )
            for idx in range(len(allhistory)):
                onenetoutpath = netoutpath + "/network" + str(idx)
                with open(onenetoutpath, "wb") as f:
                    pickle.dump(allhistory[idx][1], f)
            hvdata = pd.Series(self.hvhistory)
            hvdata.to_csv(outpath + "/" + outpath + "hv_history.csv")
            outnetnumdata = pd.Series(self.outnetworknum_history)
            outnetnumdata.to_csv(outpath + "/" + outpath + "outnetnum_history.csv")
            visualize.plot_stats2D(
                len(self.most_fit_genomes),
                self.outnetworknum_history,
                ylog=False,
                view=False,
                ylabel="Output networknum",
                filename="{}/Outnetnum.png".format(outpath),
                title="Output networknum",
            )
            visualize.plot_stats2D(
                len(self.most_fit_genomes),
                self.hvhistory,
                ylog=False,
                view=False,
                ylabel="Hypervolume",
                filename="{}/Hypervolume.png".format(outpath),
                title="HyperVolume",
            )

    def post_evaluate(self, config, population, species, best_genome):
        self.most_fit_genomes.append(copy.deepcopy(best_genome))

        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        # species_cross_validation_stats = {}
        for sid, s in iteritems(species.species):
            species_stats[sid] = dict((k, v.fitness) for k, v in iteritems(s.members))
            ##species_cross_validation_stats[sid] = dict((k, v.cross_fitness) for
        ##                                                       k, v in iteritems(s.members))
        self.generation_statistics.append(species_stats)
        # self.generation_cross_validation_statistics.append(species_cross_validation_stats)
        self.get_species_now_all_fitness()

    def get_fitness_stat(self, f):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat

    def get_fitness_mean(self):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(mean)

    def get_fitness_stdev(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev)

    def get_fitness_median(self):
        """Get the per-generation median fitness."""
        return self.get_fitness_stat(median2)

    def get_average_cross_validation_fitness(self):  # pragma: no cover
        """Get the per-generation average cross_validation fitness."""
        avg_cross_validation_fitness = []
        for stats in self.generation_cross_validation_statistics:
            scores = []
            for fitness in stats.values():
                scores.extend(fitness)
            avg_cross_validation_fitness.append(mean(scores))

        return avg_cross_validation_fitness

    def best_unique_genomes(self, n):
        """Returns the most n fit genomes, with no duplication."""
        best_unique = {}
        for g in self.most_fit_genomes:
            best_unique[g.key] = g
        best_unique_list = list(best_unique.values())

        def key(genome):
            return genome.fitness

        return sorted(best_unique_list, key=key, reverse=True)[:n]

    def best_genomes(self, n):
        """Returns the n most fit genomes ever seen."""

        def key(g):
            return g.fitness

        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self):
        """Returns the most fit genome ever seen."""
        return self.best_genomes(1)[0]

    def save(self):
        self.save_genome_fitness()
        self.save_species_count()
        self.save_species_fitness()

    def save_genome_fitness(
        self, delimiter=" ", filename="fitness_history.csv", with_cross_validation=False
    ):
        """ Saves the population's best and average fitness. """
        with open(filename, "w") as f:
            w = csv.writer(f, delimiter=delimiter)

            best_fitness = [c.fitness for c in self.most_fit_genomes]
            avg_fitness = self.get_fitness_mean()

            if with_cross_validation:  # pragma: no cover
                cv_best_fitness = [c.cross_fitness for c in self.most_fit_genomes]
                cv_avg_fitness = self.get_average_cross_validation_fitness()
                for best, avg, cv_best, cv_avg in zip(
                    best_fitness, avg_fitness, cv_best_fitness, cv_avg_fitness
                ):
                    w.writerow([best, avg, cv_best, cv_avg])
            else:
                for best, avg in zip(best_fitness, avg_fitness):
                    w.writerow([best, avg])

    def save_species_count(self, delimiter=" ", filename="speciation.csv"):
        """ Log speciation throughout evolution. """
        with open(filename, "w") as f:
            w = csv.writer(f, delimiter=delimiter)
            for s in self.get_species_sizes():
                w.writerow(s)

    def save_species_fitness(
        self, delimiter=" ", null_value="NA", filename="species_fitness.csv"
    ):
        """ Log species' average fitness throughout evolution. """
        with open(filename, "w") as f:
            w = csv.writer(f, delimiter=delimiter)
            for s in self.get_species_fitness(null_value):
                w.writerow(s)

    def get_species_sizes(self):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_counts = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts

    def get_species_fitness(self, null_value=""):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_fitness = []
        for gen_data in self.generation_statistics:
            member_fitness = [
                gen_data.get(sid, []) for sid in range(1, max_species + 1)
            ]
            fitness = []
            for mf in member_fitness:
                if mf:
                    fitness.append(mean(mf))
                else:
                    fitness.append(null_value)
            species_fitness.append(fitness)

        return species_fitness

    def get_species_now_all_fitness(self, null_value=""):
        self.species_now_all_fitness = []
        now_all_species = set()
        now_all_species = now_all_species.union(self.generation_statistics[-1].keys())
        # for gen_data in self.generation_statistics:
        #     all_species = all_species.union(gen_data.keys())

        max_species = max(now_all_species)
        species_fitness = []
        last_gen_data = self.generation_statistics[-1]
        # for gen_data in self.generation_statistics:
        #     member_fitness = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
        #     fitness = []
        #     for mf in member_fitness:
        #         if mf:
        #             fitness.append(mf)
        #         else:
        #             fitness.append(null_value)
        #     species_fitness.append(fitness)
        member_fitness = [
            last_gen_data.get(sid, []) for sid in range(1, max_species + 1)
        ]
        fitness = []
        for mf in member_fitness:
            if mf:
                fitness.append(mf)
            # else:
            #    fitness.append(null_value)
        species_fitness.append(fitness)
        self.species_now_all_fitness = species_fitness
        return self.species_now_all_fitness
