import matplotlib.pyplot as plt
import sys, operator
import os
import time
import math
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import argparse
import threading
import copy

from matplotlib.patches import Ellipse, Polygon
from numpy import nan

def some_Stuff():
    #Set pattern array
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

    font = {'size'   : 24}

    plt.rc('font', **font)

    N = 3

    Types = 1
    ind = np.arange(N)  # the x locations for the groups
    width = 1.0/ (Types + 2)     # the width of the bars

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .45*minsize/xsize
    ylim = .45*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

    #fig, ax = plt.subplots()


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


if __name__ == "__main__":
    with open('BusTrackerQueryProgression.csv') as f:
        lines = f.readlines()
        header = lines[0]
        tokens = header.split(",") # last col is labels
        cols = {}
        for token in tokens:
            cols[token] = []
        for j in range(2,48):
            line = lines[j]
            lineToks = line.split(",")
            assert len(lineToks) == len(cols)
            for i in range(len(lineToks)):
                if lineToks[i] != '':
                    cols[tokens[i]].append(float(lineToks[i]))
    #t1 = np.arange(0.0, 5.0, 0.1)
    #t2 = np.arange(0.0, 5.0, 0.02)
    fig = plt.figure()
    #adjustFigAspect(fig,aspect=1)
    #plt.subplots_adjust(top=0.88)
    some_Stuff()
    #plt.plot(cols[tokens[6]], cols[tokens[0]], 'r', linewidth=2, marker = 'x', label = 'QBC(2)')
    #plt.plot(cols[tokens[6]], cols[tokens[1]], 'g', linewidth = 2, marker = '+', label= 'Margin')
    #plt.plot(cols[tokens[6]], cols[tokens[0]], 'r--', linewidth = 2, label = 'createQBC(2)')
    #plt.plot(cols[tokens[6]], cols[tokens[1]], 'b--', linewidth = 2, label = 'createQBC(20)')
    #plt.plot(cols[tokens[4]], cols[tokens[0]], 'm', linewidth = 4, marker = '^', markevery=15, markersize = 16, label= '#DELETE')
    #plt.plot(cols[tokens[4]], cols[tokens[1]], 'r', linewidth = 4, marker = '*', markevery=15, markersize = 16, label= '#INSERT')
    #plt.plot(cols[tokens[4]], cols[tokens[2]], 'g', linewidth = 4, marker = 'o', markevery=15, markersize = 16, label= '#SELECT')
    plt.plot(cols[tokens[1]], cols[tokens[0]], 'b', linewidth = 4, marker = 'v', markerfacecolor='red', markevery=1, markersize = 16)
    
    plt.xticks(np.arange(2,48,5))
    plt.xlabel('QueryID',fontsize=38)
    #plt.yticks(np.arange(0, 1.1, 0.1))
    #plt.yscale('log')
    plt.ylabel('Starting Episode ID',fontsize=38)
    plt.title("Query Progression - Starting Episode vs. QueryID\n(Bus Tracker)", fontsize=38)
    #plt.legend(loc = 'lower left', prop={'size':22}) 
    #plt.plot([1,2,3,4],[5,6,7,8], 'ro')
    #plt.savefig('BusTrackerQueryProgression.png')
    plt.show()

