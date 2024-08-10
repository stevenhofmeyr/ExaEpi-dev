#!/usr/bin/env python

import sys
import time
import argparse
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np


def latlng_to_grid(agent1, agent2, lat, lng):
    lat1 = agent1['x-position']
    lng1 = agent1['y-position']
    x1 = int(agent1['home'].split(',')[0])
    y1 = int(agent1['home'].split(',')[1])
    lat2 = agent2['x-position']
    lng2 = agent2['y-position']
    x2 = int(agent2['home'].split(',')[0])
    y2 = int(agent2['home'].split(',')[1])
    x_scale = abs(x1 - x2) / abs(lat1 - lat2)
    y_scale = abs(y1 - y2) / abs(lng1 - lng2)
    min_lat = min(df['x-position'])
    min_lng = min(df['y-position'])
    return int(round((lat - min_lat) * x_scale) + 1), int(round((lng - min_lng) * y_scale))


def plot_population(df):
    px = 1.0 / plt.rcParams['figure.dpi']
    plt.subplots(figsize=(1920*px, 1200*px))

    plot_grid = False
    if plot_grid == True:
        df_home_locs = df[['home']]
        df_home_locs.insert(0, 'ones', int(1))
        df_home_counts = df_home_locs.groupby(['home']).size()
        y = [int(s.split(',')[0]) for s in list(df_home_counts.index)]
        x = [int(s.split(',')[1]) for s in list(df_home_counts.index)]
        plt.xlabel('Grid x')
        plt.ylabel('Grid y')
    else:
        df_home_locs = df[['x-position', 'y-position']].copy()
        df_home_locs.insert(0, 'ones', int(1))
        df_home_counts = df_home_locs.groupby(['x-position', 'y-position']).size()
        y = [s[0] for s in list(df_home_counts.index)]
        x = [s[1] for s in list(df_home_counts.index)]
        #plt.xlabel('Longitude')
        #plt.ylabel('Latitude')

    z = list(df_home_counts / 20)

    plt.scatter(x, y, s=z, color='blue', alpha=0.3)

    plot_cities = True
    if plot_cities == True:
         # read in nm cities
         df_cities = pd.read_csv("nm-cities.csv")
         df_large_cities = df_cities[df_cities['population'] > 6000][['city', 'lat', 'lng', 'population']]
         print(df_large_cities)

         for index, entry in df_large_cities.iterrows():
            if entry.city not in ['North Valley', 'South Valley', 'Rio Rancho', 'Bloomfield', 'Santa Teresa', 'Chaparral']:
               if plot_grid == True:
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

    plt.tight_layout()
    plt.savefig(args.files[0] + "-popluation.pdf")


if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser(description="Plot agents file")
    parser.add_argument("--files", "-f", required=True, nargs="+", help="Agent csv files")
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

    plot_population(df)
