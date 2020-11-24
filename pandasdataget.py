import datetime
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def plot_stats2D(generation, means=None, stds=None, mins=None, quartile_25=None, quartile_50=None, quartile_75=None, maxs=None, ylog=False, view=False, ylabel='Hypervolume',filename='plot_name.svg',title='plot_title', ymin=0, ymax=100):
    """ Plots the population's species pareto front. """
    plt.figure(figsize=(5,5))
    plt.rcParams["font.size"] = 12
    plot_x = range(1,generation+1)
    #plot_x = plot_x+1
    if(means!=None):
        means_y = np.array(means)
        plt.plot(plot_x, means_y, 'g-', label="average")
    if(stds!=None):
        stds_y = np.array(stds)
        plt.plot(plot_x, means_y+stds_y, 'b-.',label="+1 std")
        plt.plot(plot_x, means_y-stds_y, 'b-.',label="-1 std")
    if(mins!=None):
        mins_y = np.array(mins)
        plt.plot(plot_x, mins_y, 'r-',label="mins")
    if(quartile_25!=None):
        quartile_25_y = np.array(quartile_25)
        plt.plot(plot_x, quartile_25_y, 'g:',label="25%")
    if(quartile_50!=None):
        quartile_50_y = np.array(quartile_50)
        plt.plot(plot_x, quartile_50_y, 'g:',label="50%")
    if(quartile_75!=None):
        quartile_75_y = np.array(quartile_75)
        plt.plot(plot_x, quartile_75_y, 'g:',label="75%")
    if(maxs!=None):
        maxs_y = np.array(maxs)
        plt.plot(plot_x, maxs_y, 'r-',label="maxs")
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, max(plot_x)+1, 15))
    #plt.ylim(min(mins_y),max(maxs_y))
    plt.ylim(ymin,ymax)
    #plt.legend(loc="best")
    plt.legend(loc="lower left")
    if ylog:
        plt.gca().set_yscale('symlog')
    print("Saving " +str(filename))
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.95)
    #plt.tight_layout()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    all_data = None
    
    parser.add_argument('momethod', help='multi-optimization method')
    parser.add_argument('usenum', help='using data num')
    parser.add_argument('generation', help='generation num(x axis)')
    parser.add_argument('dataname',help='using data kind')
    parser.add_argument('--path',default='./' ,help='directory path which exists datas')
    args = parser.parse_args()
    method = args.momethod
    datanum = int(args.usenum)
    dataname = args.dataname
    generation = int(args.generation)
    path = args.path
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path,f))]
    files_dir = [f for f in files_dir if f.startswith(method)]
    hv_historys = []
    for dirpath in files_dir:
        files = os.listdir(dirpath)
        files_indir = [f for f in files if os.path.isfile(os.path.join(dirpath,f))]
        #print(files_indir)
        hv_history = [f for f in files_indir if f.endswith(str(dataname)+'.csv')]
        if(len(hv_history)!=0):
            hv_historys.append(dirpath+'/'+hv_history[0])
    #hv_historys = [f for f in hv_historys if f.endswith('hv_history.csv')]
    #print(files_dir)
    #print(hv_historys)
    column=0
    while len(hv_historys)>0:
        all_data = pd.read_csv(hv_historys.pop(0),index_col=0)
        if(len(all_data)==generation):
            datanum-=1
            break
    while(datanum>0 and len(hv_historys)>0):
        next_data = pd.read_csv(hv_historys.pop(0),index_col=0)
        next_data = next_data.rename(columns={'0':str(column)})
        #print(len(next_data))
        #print(next_data)
        #all_data.append(next_data)
        if(len(next_data)==generation):
            column+=1
            datanum-=1
            all_data = pd.concat([all_data, next_data], axis=1)
    #print(all_data)
    all_describe = all_data.T.describe()
    #print(all_describe)
    means = []
    stds = []
    quartile_0 = []
    quartile_25 = []
    quartile_50 = []
    quartile_75 = []
    quartile_100 = []
    for i in range(generation):
        means.append(all_describe[i]['mean'])
        stds.append(all_describe[i]['std'])
        quartile_0.append(all_describe[i]['min'])
        quartile_25.append(all_describe[i]['25%'])
        quartile_50.append(all_describe[i]['50%'])
        quartile_75.append(all_describe[i]['75%'])
        quartile_100.append(all_describe[i]['max'])
    """
    print(means)
    print(stds)
    print(quartile_0)
    print(quartile_25)
    print(quartile_50)
    print(quartile_75)
    print(quartile_100)
    """
    #print(method)
    #print(datanum)
    #print(dataname)
    #print(generation)
    if dataname=="hv_history":
        plot_stats2D(generation, means=means, stds=stds, mins=quartile_0, quartile_25=quartile_25, quartile_50=quartile_50, quartile_75=quartile_75, maxs=quartile_100, ylog=False, view=False, ylabel='Hypervolume',filename=str(len(all_data.T))+'_trials_'+str(method)+'_HyperVolume.png',title=str(method) + ' ' + str(len(all_data.T)) +' trials Hypervolume describe', ymin=0, ymax=1.333)
    elif dataname=="gd_history":
        plot_stats2D(generation, means=means, stds=stds, mins=quartile_0, quartile_25=quartile_25, quartile_50=quartile_50, quartile_75=quartile_75, maxs=quartile_100, ylog=False, view=False, ylabel='Genetic distance',filename=str(len(all_data.T))+'_trials_'+str(method)+'_Genedistance.png',title=str(method) + ' ' + str(len(all_data.T)) +' trials Genetic distance describe', ymin=1.25, ymax=3.5)

if __name__=='__main__':
    main()

        
