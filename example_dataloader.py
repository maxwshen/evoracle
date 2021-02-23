'''
  Loads data from /example_data/ into Python objects to run Evoracle.
'''
import pandas as pd
import pickle


def load_obs_reads_df(inp_fn):
  # Loads csv
  return pd.read_csv(inp_fn)


def load_proposed_gts(inp_fn):
  # Loads newline-delimited list of full-length genotypes
  with open(inp_fn) as f:
    lines = f.readlines()
  return [s.strip() for s in lines]


def load_read_segments(inp_fn):
  '''
    Parses a text file where each line is a space-delimited list of integers, representing mutation positions.
  '''
  res = []
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if line:
        res.append([int(s) for s in line.split()])
  return res