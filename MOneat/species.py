"""Divides the population into species based on genomic distances."""
from itertools import count

from MOneat.math_util import mean, stdev
from MOneat.six_util import iteritems, iterkeys, itervalues
from MOneat.config import ConfigParameter, DefaultClassConfig

class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.age = 0
        self.representative = None
        self.members = {}
        self.fitness = None
        self.best_fitness = None
        self.single_best_fitnesses = None
        self.adjusted_fitness = None
        self.fitness_history = []
        self.ref_points = None
        self.best_ref_points = None
        self.before_ref_points = None
        self.ref_points_history = []
        self.fitness_weight = None
        self.priority_fitness = None
        self.priority_fitness_history = []
        self.temperature = None
        self.coolrate = None
        self.saperiod = 1

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]

    def get_priority_fitnesses(self):
        return [m.priority_fitness for m in itervalues(self.members)]

class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d

class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float),
                                   ConfigParameter('compatibility_fwthreshold',float,-1.0),
                                   ConfigParameter('coolrate',float,1.0),
                                   ConfigParameter('initialsarate', float, 1.00),
                                   ConfigParameter('saperiod',int,3),
                                   ConfigParameter('speciesnummax',int,10)])

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold
        compatibility_fwthreshold = self.species_set_config.compatibility_fwthreshold
        temperature = self.species_set_config.initialsarate
        coolrate = self.species_set_config.coolrate
        saperiod = self.species_set_config.saperiod


        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = []
            s.age = generation - s.created
            if s.temperature is None or s.coolrate is None:
                s.temperature = temperature
                s.coolrate = coolrate
                s.saperiod = saperiod
            else:
                if s.age%saperiod==0:
                    s.temperature*=s.coolrate
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            #print(new_rep)
            if(compatibility_fwthreshold>0):
                s.fitness_weight=new_rep.fitness.copy()
                mx = max(s.fitness_weight)
                for idx in range(len(s.fitness_weight)):
                    s.fitness_weight[idx]/=mx
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                #if(d<7.0):
                #    print("species sid:{0}, rid:{1},distance with g:{2}".format(sid,rid,d))
                if d < compatibility_threshold or len(new_representatives)>=self.species_set_config.speciesnummax:
                    candidates.append((d, sid))

            if candidates:
                if(compatibility_fwthreshold<0 or generation==0):
                    ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                    new_members[sid].append(gid)
                else:
                    candidates.sort(key=lambda x: x[0])
                    append_newmember=False
                    #print(candidates)
                    for ignored_sdist, sid in candidates:
                        #print(ignored_sdist, sid)
                        candspecies = self.species[sid]
                        speciesfw = candspecies.fitness_weight
                        gfw=g.fitness.copy()
                        mx = max(gfw)
                        fwdistance = 0.0
                        for idx in range(len(gfw)):
                            gfw[idx]/=mx
                            fwdistance+=abs(gfw[idx]-speciesfw[idx])
                        if(fwdistance< compatibility_fwthreshold):
                            new_members[sid].append(gid)
                            append_newmember=True
                            break
                    if(append_newmember==False):
                        ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                        new_members[sid].append(gid)

            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.genetic_distance(config,gdmean)
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
