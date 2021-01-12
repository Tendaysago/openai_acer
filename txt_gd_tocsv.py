import datetime
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import re

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
    if means != None:
        means_y = np.array(means)
        plt.plot(plot_x, means_y, "g-", label="average")
    if stds != None:
        stds_y = np.array(stds)
        plt.plot(plot_x, means_y + stds_y, "b-.", label="+1 std")
        plt.plot(plot_x, means_y - stds_y, "b-.", label="-1 std")
    if mins != None:
        mins_y = np.array(mins)
        plt.plot(plot_x, mins_y, "r-", label="mins")
    if quartile_25 != None:
        quartile_25_y = np.array(quartile_25)
        plt.plot(plot_x, quartile_25_y, "g:", label="25%")
    if quartile_50 != None:
        quartile_50_y = np.array(quartile_50)
        plt.plot(plot_x, quartile_50_y, "g:", label="50%")
    if quartile_75 != None:
        quartile_75_y = np.array(quartile_75)
        plt.plot(plot_x, quartile_75_y, "g:", label="75%")
    if maxs != None:
        maxs_y = np.array(maxs)
        plt.plot(plot_x, maxs_y, "r-", label="maxs")
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, max(plot_x) + 1, 15))
    plt.ylim(ymin, ymax)
    # plt.legend(loc="best")
    plt.legend(loc="lower left")
    if ylog:
        plt.gca().set_yscale("symlog")
    print("Saving " + str(filename))
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.1, top=0.95)
    # plt.tight_layout()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("momethod", help="multi-optimization method")
    parser.add_argument("usenum", help="using data num")
    parser.add_argument("generation", help="generation num(x axis)")
    parser.add_argument("--dataname", default="log.txt", help="using data kind")
    parser.add_argument(
        "--path", default="./", help="directory path which exists datas"
    )
    args = parser.parse_args()
    method = args.momethod
    datanum = int(args.usenum)
    dataname = args.dataname
    generation = int(args.generation)
    path = args.path
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    files_dir = [f for f in files_dir if f.startswith(method)]
    log_historys = []
    for dirpath in files_dir:
        files = os.listdir(dirpath)
        files_indir = [f for f in files if os.path.isfile(os.path.join(dirpath, f))]
        # print(files_indir)
        log_history = [f for f in files_indir if f.endswith(str(dataname))]
        if len(log_history) != 0:
            log_historys.append(dirpath + "/" + log_history[0])
    speciesnum_historys = []
    populationsnum_historys = []
    all_speciesnum_data = None
    all_populationsnum_data = None
    column = -1
    while len(log_historys) > 0 and datanum > 0:
        log = log_historys.pop(0)
        speciesnum_list = []
        populationnum_list = []
        with open(log) as f:
            speciesnum = 0
            generationnum = 0
            populationnum = 0
            for s_line in f:
                # print(s_line)
                if "******" in s_line:
                    generationnum = int(re.sub("\D", "", s_line)) + 1
                if (
                    "members" in s_line
                    and "Population" in s_line
                    and "species" in s_line
                ):
                    matchpattern = re.sub("\D", "", s_line)
                    # print(matchpattern)
                    populationnum, speciesnum = int(matchpattern[0:3]), int(
                        matchpattern[3:]
                    )
                if "hypervolume" in s_line:
                    speciesnum_list.append(speciesnum)
                    populationnum_list.append(populationnum)
        if generationnum == generation:
            datanum -= 1
            speciesnumdata = pd.Series(speciesnum_list)
            populationnumdata = pd.Series(populationnum_list)
            speciesnumdata.to_csv(
                str(method)
                + "speciesnum_history"
                + str(int(args.usenum) - datanum)
                + ".csv"
            )
            speciesnum_historys.append(
                str(method)
                + "speciesnum_history"
                + str(int(args.usenum) - datanum)
                + ".csv"
            )
            populationnumdata.to_csv(
                str(method)
                + "populationnum_history"
                + str(int(args.usenum) - datanum)
                + ".csv"
            )
            populationsnum_historys.append(
                str(method)
                + "populationnum_history"
                + str(int(args.usenum) - datanum)
                + ".csv"
            )

    while len(speciesnum_historys) > 0:
        if column == -1:
            column = 0
            all_speciesnum_data = pd.read_csv(speciesnum_historys.pop(0), index_col=0)
            all_populationsnum_data = pd.read_csv(
                populationsnum_historys.pop(0), index_col=0
            )
        else:
            column += 1
            next_speciesnum_data = pd.read_csv(speciesnum_historys.pop(0), index_col=0)
            next_speciesnum_data = next_speciesnum_data.rename(
                columns={"0": str(column)}
            )
            next_populationsnum_data = pd.read_csv(
                populationsnum_historys.pop(0), index_col=0
            )
            next_populationsnum_data = next_populationsnum_data.rename(
                columns={"0": str(column)}
            )
            all_speciesnum_data = pd.concat(
                [all_speciesnum_data, next_speciesnum_data], axis=1
            )
            all_populationsnum_data = pd.concat(
                [all_populationsnum_data, next_populationsnum_data], axis=1
            )

    all_species_describe = all_speciesnum_data.T.describe()
    all_populations_describe = all_populationsnum_data.T.describe()
    # print(all_describe)
    species_means = []
    populations_means = []
    species_stds = []
    populations_stds = []
    species_quartile_0 = []
    populations_quartile_0 = []
    species_quartile_25 = []
    populations_quartile_25 = []
    species_quartile_50 = []
    populations_quartile_50 = []
    species_quartile_75 = []
    populations_quartile_75 = []
    species_quartile_100 = []
    populations_quartile_100 = []
    for i in range(generation):
        species_means.append(all_species_describe[i]["mean"])
        populations_means.append(all_populations_describe[i]["mean"])
        species_stds.append(all_species_describe[i]["std"])
        populations_stds.append(all_populations_describe[i]["std"])
        species_quartile_0.append(all_species_describe[i]["min"])
        populations_quartile_0.append(all_populations_describe[i]["min"])
        species_quartile_25.append(all_species_describe[i]["25%"])
        populations_quartile_25.append(all_populations_describe[i]["25%"])
        species_quartile_50.append(all_species_describe[i]["50%"])
        populations_quartile_50.append(all_populations_describe[i]["50%"])
        species_quartile_75.append(all_species_describe[i]["75%"])
        populations_quartile_75.append(all_populations_describe[i]["75%"])
        species_quartile_100.append(all_species_describe[i]["max"])
        populations_quartile_100.append(all_populations_describe[i]["max"])
    """
    print(means)
    print(stds)
    print(quartile_0)
    print(quartile_25)
    print(quartile_50)
    print(quartile_75)
    print(quartile_100)
    """
    plot_stats2D(
        generation,
        means=species_means,
        stds=species_stds,
        mins=species_quartile_0,
        quartile_25=species_quartile_25,
        quartile_50=species_quartile_50,
        quartile_75=species_quartile_75,
        maxs=species_quartile_100,
        ylog=False,
        view=False,
        ylabel="Species num",
        filename=str(len(all_speciesnum_data.T))
        + "_trials_"
        + str(method)
        + "_speciesnum.png",
        title=str(method)
        + " "
        + str(len(all_speciesnum_data.T))
        + " trials Species num describe",
        ymin=0,
        ymax=10,
    )
    plot_stats2D(
        generation,
        means=populations_means,
        stds=populations_stds,
        mins=populations_quartile_0,
        quartile_25=populations_quartile_25,
        quartile_50=populations_quartile_50,
        quartile_75=populations_quartile_75,
        maxs=populations_quartile_100,
        ylog=False,
        view=False,
        ylabel="Populations num",
        filename=str(len(all_populationsnum_data.T))
        + "_trials_"
        + str(method)
        + "_populationsnum.png",
        title=str(method)
        + " "
        + str(len(all_populationsnum_data.T))
        + " trials Populations num describe",
        ymin=190,
        ymax=400,
    )
    # print(all_data)

    """
    plot_stats2D(generation, means=means, stds=stds, mins=quartile_0, quartile_25=quartile_25, quartile_50=quartile_50, quartile_75=quartile_75, maxs=quartile_100, ylog=False, view=False, ylabel='Hypervolume',filename=str(len(all_data.T))+'_trials_'+str(method)+'_HyperVolume.png',title=str(method) + ' ' + str(len(all_data.T)) +' trials Hypervolume describe')
    """


if __name__ == "__main__":
    main()
