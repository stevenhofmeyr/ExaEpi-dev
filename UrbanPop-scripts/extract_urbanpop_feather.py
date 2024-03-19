#!/usr/bin/env python

import sys
import os.path
import pandas

fname = sys.argv[1]
print("Reading data from", fname)
df = pandas.read_feather(fname)
fname, extension = os.path.splitext(fname)
fname = fname + ".csv"
print("Writing data to", fname)
df.to_csv(fname)
