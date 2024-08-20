#!/usr/bin/env python

import sys
import time
import argparse
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
import math
import functools

x_scale = 0
y_scale = 0
min_lat = 0
min_lng = 0

def latlng_to_grid(agent1, agent2, lat, lng):
    global x_scale
    global y_scale
    global min_lat
    global min_lng
    lat1 = agent1['x-position']
    lng1 = agent1['y-position']
    x1 = int(agent1['home'].split(',')[0])
    y1 = int(agent1['home'].split(',')[1])
    lat2 = agent2['x-position']
    lng2 = agent2['y-position']
    x2 = int(agent2['home'].split(',')[0])
    y2 = int(agent2['home'].split(',')[1])
    if x_scale == 0:
        x_scale = abs(x1 - x2) / abs(lat1 - lat2)
        y_scale = abs(y1 - y2) / abs(lng1 - lng2)
        min_lat = min(df['x-position'])
        min_lng = min(df['y-position'])
    return int(round((lat - min_lat) * x_scale) + 1), int(round((lng - min_lng) * y_scale))


def plot_cities(realcoords):
    # read in nm cities
    df_cities = pd.read_csv("nm-cities.csv")
    df_large_cities = df_cities[df_cities['population'] > 6000][['city', 'lat', 'lng', 'population']]
    #print(df_large_cities)

    for _, entry in df_large_cities.iterrows():
        if entry.city not in ['North Valley', 'South Valley', 'Rio Rancho', 'Bloomfield', 'Santa Teresa', 'Chaparral']:
            if realcoords == False:
                shift_size = 3
                y, x = latlng_to_grid(df.iloc[0], df.iloc[-1], entry.lat, entry.lng)
            else:
                shift_size = 0.005
                y = entry.lat
                x = entry.lng
            # trick for creating outlined text
            for shift in [(0, -shift_size), (0, shift_size), (-shift_size, 0), (shift_size, 0)]:
                plt.text(x + shift[0], y + shift[1], entry.city, color='white', weight='bold', variant='small-caps',
                        ha='center', va='center')
            plt.text(x, y, entry.city, color='black', weight='bold', variant='small-caps', ha='center', va='center')


def plot_population(df, realcoords, worklocs, citylocs):
    px = 1.0 / plt.rcParams['figure.dpi']
    _, ax = plt.subplots(figsize=(1920*px, 1200*px))

    if realcoords == False:
        if worklocs == True:
           df = df[(df['workg'] != 0)]
           col = 'work'
        else:
           col = 'home'
        df_locs = df[[col]]
        df_locs.insert(0, 'ones', int(1))
        df_counts = df_locs.groupby([col]).size()
        # sorting ensures higher values show up over lower values
        df_counts.sort_values(inplace=True)
        y = [int(s.split(',')[0]) for s in list(df_counts.index)]
        x = [int(s.split(',')[1]) for s in list(df_counts.index)]
        plt.xlabel('Grid x')
        plt.ylabel('Grid y')
    else:
        df_home_locs = df[['x-position', 'y-position']].copy()
        df_home_locs.insert(0, 'ones', int(1))
        df_counts = df_home_locs.groupby(['x-position', 'y-position']).size()
        y = [s[0] for s in list(df_counts.index)]
        x = [s[1] for s in list(df_counts.index)]
        plt.xlabel('Particle position x')
        plt.ylabel('Particle position y')

    print("Plotting", len(x), "locations, with total count", sum(list(df_counts)))

    cmap = mp.colormaps['plasma']
    #cmap = mp.colormaps['copper']
    #cmap = mp.colormaps['cool']
    max_count = 50000 #max(df_counts)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    z = list(np.log(df_counts) / np.log(max_count))
    print("min x", min(x), "max x", max(x), "min y", min(y), "max y", max(y), "min z", min(z), "max z", max(z))
    plt.scatter(x, y, s=20, color=cmap(z),  alpha=1.0)
    if citylocs:
        plot_cities(realcoords)
    plt.tight_layout()
    out_fname = args.files[0] + "-population"
    if worklocs == True:
        out_fname += "-work"
    else:
        out_fname += "-home"
    plt.savefig(out_fname + ".png")


def plot_infected(df, citylocs):
    px = 1.0 / plt.rcParams['figure.dpi']
    _, ax = plt.subplots(figsize=(1920*px, 1200*px))

    y = [int(s.split(',')[0]) for s in list(df['home'])]
    x = [int(s.split(',')[1]) for s in list(df['home'])]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    print("min x", min_x, "max x", max_x, "min y", min_y, "max y", max_y)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    df_counts = df.loc[df['status'] == 1].groupby('home')['status'].sum()
    print(df_counts)
    # sorting ensures higher values show up over lower values
    df_counts.sort_values(inplace=True)
    y = [int(s.split(',')[0]) for s in list(df_counts.index)]
    x = [int(s.split(',')[1]) for s in list(df_counts.index)]
    plt.xlabel('Grid x')
    plt.ylabel('Grid y')

    print("Plotting", len(x), "locations, with total count", sum(list(df_counts)))

    cmap = mp.colormaps['plasma']
    #cmap = mp.colormaps['copper']
    #cmap = mp.colormaps['cool']
    max_count = 5050 #max(df_counts)
    print("max count", max_count)
    z = list(np.log(df_counts) / np.log(max_count))
    print("min x", min(x), "max x", max(x), "min y", min(y), "max y", max(y), "min z", min(z), "max z", max(z))
    plt.scatter(x, y, s=20, color=cmap(z),  alpha=1.0)
    if citylocs:
        plot_cities(realcoords=False);
    plt.tight_layout()
    out_fname = args.files[0] + "-infected"
    plt.savefig(out_fname + ".png")



def update_func(frame_i, ax, scatter):
    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.random.rand(10)
    scatter.set_offsets(np.c_[x, y])
    scatter.set_color(mp.colormaps['plasma'](z))
    ax.set_title("Frame " + str(frame_i))
    return scatter,


if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser(description="Plot agents file")
    parser.add_argument("--files", "-f", required=True, nargs="+", help="Agent csv files")
    parser.add_argument("--realcoords", "-r", action='store_true', help="Plot real underlying coordinates rather than grid")
    parser.add_argument("--worklocs", "-w", action='store_true', help="Plot work grid locations rather than home")
    parser.add_argument("--citylocs", "-c", action='store_true', help="Plot New Mexico major cities")
    parser.add_argument("--infected", "-i", action='store_true', help="Plot infected")
    args = parser.parse_args()

    dfs = []
    for fname in args.files:
        print("Reading data from", fname)
        t = time.time()
        dfs.append(pd.read_csv(fname, sep='\t'))
        print("Read", len(dfs[-1].index), "records in %.3f s" % (time.time() - t))
    df = pd.concat(dfs)

    print("Loaded", len(args.files), "files in %.2f s" % (time.time() - t))
    print(df)

    if args.infected:
        plot_infected(df, args.citylocs)
        if False:
            fig, ax = plt.subplots()
            x = np.random.rand(10)
            y = np.random.rand(10)
            z = np.random.rand(10)
            scatter = ax.scatter(x, y, color=mp.colormaps['plasma'](z))
            x = np.random.rand(10)
            y = np.random.rand(10)
            animation = mp.animation.FuncAnimation(fig, functools.partial(update_func, ax=ax, scatter=scatter), frames=50, interval=500)
            plt.show()

    else:
        plot_population(df, args.realcoords, args.worklocs, args.citylocs)

    # now do new plot of worker flows


