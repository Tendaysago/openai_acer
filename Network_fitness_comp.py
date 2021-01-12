import datetime
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import pickle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


def plot_stats2D(
    generation,
    means=None,
    stds=None,
    mins=None,
    quartile_25=None,
    quartile_50=None,
    quartile_75=None,
    maxs=None,
    ylog=False,
    view=False,
    ylabel="Hypervolume",
    filename="plot_name.svg",
    title="plot_title",
    ymin=0,
    ymax=100,
):
    """ Plots the population's species pareto front. """
    plt.figure(figsize=(5, 5))
    plt.rcParams["font.size"] = 12
    plot_x = range(1, generation + 1)
    # plot_x = plot_x+1
    if means is None:
        means_y = np.array(means)
        plt.plot(plot_x, means_y, "g-", label="average")
    if stds is None:
        stds_y = np.array(stds)
        plt.plot(plot_x, means_y + stds_y, "b-.", label="+1 std")
        plt.plot(plot_x, means_y - stds_y, "b-.", label="-1 std")
    if mins is None:
        mins_y = np.array(mins)
        plt.plot(plot_x, mins_y, "r-", label="mins")
    if quartile_25 is None:
        quartile_25_y = np.array(quartile_25)
        plt.plot(plot_x, quartile_25_y, "g:", label="25%")
    if quartile_50 is None:
        quartile_50_y = np.array(quartile_50)
        plt.plot(plot_x, quartile_50_y, "g:", label="50%")
    if quartile_75 is None:
        quartile_75_y = np.array(quartile_75)
        plt.plot(plot_x, quartile_75_y, "g:", label="75%")
    if maxs is None:
        maxs_y = np.array(maxs)
        plt.plot(plot_x, maxs_y, "r-", label="maxs")
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, max(plot_x) + 1, 15))
    # plt.ylim(min(mins_y),max(maxs_y))
    plt.ylim(ymin, ymax)
    # plt.legend(loc="best")
    plt.legend(loc="lower left")
    if ylog:
        plt.gca().set_yscale("symlog")
    print("Saving " + str(filename))
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.95)
    # plt.tight_layout()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


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


def sortNondominated2(individuals, k, first_front_only=False, reverse=False):
    if k == 0:
        return []
    fits = []
    for ind in individuals:
        fits.append(ind[1])
    current_front = []
    next_front = []
    fronts = [[]]
    current_front_indices = []
    next_front_indices = []
    fronts_indices = [[]]
    dominating_indices = [[] for _ in range(len(individuals))]
    n_dominated = np.zeros(len(individuals))
    for i in range(len(fits)):
        # print("i, fit_i: {0}, {1}".format(i,fit_i))
        # print("map_fit_ind[fit_i]: {0}".format(map_fit_ind[fit_i]))
        for j in range(i + 1, len(fits)):
            if isdominates(fits[i], fits[j]):
                n_dominated[j] += 1
                dominating_indices[i].append(j)
            elif isdominates(fits[j], fits[i]):
                n_dominated[i] += 1
                dominating_indices[j].append(i)
        if n_dominated[i] == 0:
            current_front.append(fits[i])
            current_front_indices.append(i)

    # print(current_front_indices)

    for idx in current_front_indices:
        fronts_indices[-1].append(idx)
        fronts[-1].append(tuple(individuals[idx]))
    # print(fronts_indices)
    # print(fronts)
    pareto_sorted = len(fronts[-1])

    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            for i in current_front_indices:
                for j in dominating_indices[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front_indices.append(j)
                        next_front.append(tuple(individuals[j]))
                        pareto_sorted += 1
            fronts_indices.append(next_front_indices)
            fronts.append(next_front)
            current_front_indices = next_front_indices
            current_front = next_front
            next_front = []
            next_front_indices = []
    # print("END")
    return fronts, fronts_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("momethod", help="multi-optimization method")
    parser.add_argument("usenum", help="using data num")
    parser.add_argument("generation", help="generation num(x axis)")
    parser.add_argument("dataname", help="using data kind")
    parser.add_argument(
        "--path", default="./", help="directory path which exists datas"
    )
    args = parser.parse_args()
    method = args.momethod
    datanum = int(args.usenum)
    dataname = args.dataname
    generation = int(args.generation)
    path = args.path
    method_list = ["RNSGA2", "NSGA3None", "NSGA2HV"]
    print(len(method_list))
    dirs = []
    network_dirs = []
    all_files = os.listdir(path)
    all_dirs = [f for f in all_files if os.path.isdir(os.path.join(path, f))]
    for i in range(len(method_list)):
        method_com = method_list[i]
        files = os.listdir(path)
        dirs.append([f for f in all_dirs if f.startswith(method_com)])
        network_dirs.append([])
        # print(dirs)
        for dirpath in dirs[i]:
            files = os.listdir(dirpath)
            dirs_indir = [f for f in files if os.path.isdir(os.path.join(dirpath, f))]
            # print(files_indir)
            gen_data_dir = [f for f in dirs_indir if f == "Gen_" + str(generation)]
            if len(gen_data_dir) == 1:
                network_dirs[i].append(dirpath + "/" + gen_data_dir[0] + "/Networks")
                if len(network_dirs[i]) == datanum:
                    break
        # for j in range(len(network_dirs[i])):
        #    print(network_dirs[i][j])
        # print("\n")
    all_networks = [[] for i in range(len(method_list))]
    battle_before = np.zeros((datanum, len(method_list), len(method_list)))
    battle_after = np.zeros((datanum, len(method_list), len(method_list)))
    battle_after_percent = np.zeros((datanum, len(method_list), len(method_list)))
    battle_each_percent = np.zeros((datanum, len(method_list), len(method_list)))
    battle_each_mean = np.zeros((len(method_list), len(method_list)))
    battle_after_mean = np.zeros((len(method_list), len(method_list)))
    # print(battle_results)

    for battle in range(datanum):
        for method1 in range(len(method_list)):
            for method2 in range(method1 + 1, len(method_list)):
                network_dirs1 = network_dirs[method1][battle]
                network_dirs2 = network_dirs[method2][battle]
                networks1 = os.listdir(network_dirs1)
                networks2 = os.listdir(network_dirs2)
                battle_field = []
                # print(network_dirs1)
                # print(networks1)
                # print(network_dirs2)
                # print(networks2)
                idx = 1
                for i in range(len(networks1)):
                    all_networks[method1].append(network_dirs1 + "/" + networks1[i])
                    with open(network_dirs1 + "/" + networks1[i], "rb") as f:
                        c1 = pickle.load(f)
                        battle_field.append((idx, c1.fitness))
                    idx += 1
                for i in range(len(networks2)):
                    all_networks[method2].append(network_dirs2 + "/" + networks2[i])
                    with open(network_dirs2 + "/" + networks2[i], "rb") as f:
                        c2 = pickle.load(f)
                        battle_field.append((idx, c2.fitness))
                    idx += 1
                first_front, first_idx = sortNondominated2(
                    battle_field, len(battle_field), True
                )
                # print(len(networks1))
                # print(len(networks2))
                # print(first_front)
                # print(first_idx)
                battle_before[battle][method1][method2] = len(networks1)
                battle_before[battle][method2][method1] = len(networks2)
                for front_idx in first_idx[0]:
                    if front_idx <= len(networks1):
                        battle_after[battle][method1][method2] += 1
                    else:
                        battle_after[battle][method2][method1] += 1
                battle_after_percent[battle][method1][method2] = (
                    battle_after[battle][method1][method2]
                    / battle_before[battle][method1][method2]
                    * 100.0
                )
                battle_after_mean[method1][method2] += battle_after_percent[battle][
                    method1
                ][method2]
                battle_each_percent[battle][method1][method2] = (
                    battle_after[battle][method1][method2]
                    / (
                        battle_after[battle][method1][method2]
                        + battle_after[battle][method2][method1]
                    )
                    * 100.0
                )
                battle_each_mean[method1][method2] += battle_each_percent[battle][
                    method1
                ][method2]
                battle_after_percent[battle][method2][method1] = (
                    battle_after[battle][method2][method1]
                    / battle_before[battle][method2][method1]
                    * 100.0
                )
                battle_after_mean[method2][method1] += battle_after_percent[battle][
                    method2
                ][method1]
                battle_each_percent[battle][method2][method1] = (
                    battle_after[battle][method2][method1]
                    / (
                        battle_after[battle][method1][method2]
                        + battle_after[battle][method2][method1]
                    )
                    * 100.0
                )
                battle_each_mean[method2][method1] += battle_each_percent[battle][
                    method2
                ][method1]
    battle_after_mean /= datanum
    battle_each_mean /= datanum
    print("battle_before")
    print(battle_before)
    print("battle_after")
    print(battle_after)
    print("battle_after_percent")
    print(battle_after_percent)
    print("battle_each_percent")
    print(battle_each_percent)
    print("battle_after_mean")
    print(battle_after_mean)
    print("battle_each_mean")
    print(battle_each_mean)


if __name__ == "__main__":
    main()
