'''
  Example use of Evoracle
'''

gene = 'cry1ac'
rl = 100

# Setup 
inp_dir = 'example_data/'
out_dir = f'example_evoracle_output/{gene}_{rl}nt/'

# Load data
import pandas as pd
obs_reads_df = pd.read_csv(
  inp_dir + f'{gene}_{rl}nt_obsreads.csv'
)

# Run Evoracle
import evoracle
evoracle.propose_gts_and_infer_fitness_and_frequencies(
  obs_reads_df = obs_reads_df, 
  proposed_gt_out_fn = inp_dir + f'{gene}_{rl}nt_proposedgts.txt', 
  inference_out_dir = out_dir)
