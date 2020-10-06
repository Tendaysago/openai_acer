#    This file is part of MOneat.
#
#    This graph plot implementation is based from here :
#    https://github.com/CodeReclaimers/neat-python/blob/master/examples/single-pole-balancing/visualize.py
#    Copyright (C) 2018 {Alan McIntyre and Matt Kallada and Cesar G. Miguel and Carolina Feher da Silva}

from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mpl_toolkits.mplot3d import Axes3D

colors=["red","blue","yellow","green","purple","gray","salmon","aqua",\
    "orange","lime","plum","black"]
timestamp = None

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_stats3D(generation, plot_data, ylog=False, view=False, filename='paretofront_fitness.svg'):
    """ Plots the population's species pareto front. """
    global timestamp
    if(timestamp==None):
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    fig = plt.figure(figsize=(7,7))
    ax = Axes3D(fig)
    ax.set_xlabel("Down stairs success num/10.0")
    ax.set_ylabel("Find Roomnum/5.0")
    ax.set_zlabel("Pick item num/5.0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    #generation = len(statistics.most_fit_genomes)
    #paretofront_fitness = [c.fitness for c in statistics.most_fit_genomes]
    #paretofront_fitness = statistics.species_now_all_fitness
    paretofront_fitness = plot_data
    #print(paretofront_fitness)
    #print(len(paretofront_fitness[0]))
    #print(len(paretofront_fitness[0][0]))
    #print(len(paretofront_fitness[0][0][1]))
    #assert(len(paretofront_fitness[0][0][1])==3)
    speciesnum=0
    #for sidx in range(len(paretofront_fitness[0])):
    for front in paretofront_fitness:
        #print(front)
        #if(len(colors)<=speciesnum):
        #    break
        #if(len(paretofront_fitness[0][sidx])==0):
        #    continue
        #print(paretofront_fitness[0][sidx])
        #X = np.zeros(len(paretofront_fitness[0][sidx]),dtype=float)
        #Y = np.zeros(len(paretofront_fitness[0][sidx]),dtype=float)
        #Z = np.zeros(len(paretofront_fitness[0][sidx]),dtype=float)
        X = np.zeros(len(front),dtype=float)
        Y = np.zeros(len(front),dtype=float)
        Z = np.zeros(len(front),dtype=float)
        fidx=0
        #for fitness in paretofront_fitness[0][sidx].values():
        #    X[fidx]=fitness[0]
        #    Y[fidx]=fitness[1]
        #    Z[fidx]=fitness[2]
        #    fidx+=1
        for sidx in range(len(front)):
            fitness = front[sidx]
            X[fidx]=fitness[0]
            Y[fidx]=fitness[1]
            Z[fidx]=fitness[2]
            fidx+=1
        if(len(front)!=0):
            ax.plot(X,Y,Z,marker="o", linestyle="None",color=colors[speciesnum],label="Group "+str(speciesnum+1))
            speciesnum+=1
        #print(X)
        #print(Y)
        #print(Z)
        #ax.plot(X,Y,Z,marker="o", linestyle="None",color=colors[speciesnum],label="species "+str(speciesnum+1))
        speciesnum+=1
    ax.legend(loc=2, title='species', shadow=True)
    plt.title("Generation " + str(generation) + filename)
    filename= timestamp + filename + "_gen_"+ str(generation)+".png"
    plt.savefig(filename)
    plt.close()



def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    plt.ioff()

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot