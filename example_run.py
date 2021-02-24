'''
  Example use of Evoracle
'''


# gene = 'abe8e'
gene = 'cry1ac'


# Setup 
inp_dir = 'example_data/'
out_dir = f'example_evoracle_out_{gene}/'

# Load data
import pandas as pd
obs_reads_df = pd.read_csv(
  inp_dir + f'{gene}_illumina_100nt_obsreads.csv'
)

# Run Evoracle
import evoracle
evoracle.propose_gts_and_infer_fitness_and_frequencies(
  obs_reads_df = obs_reads_df, 
  proposed_gt_out_fn = inp_dir + f'{gene}_proposedgts.txt', 
  inference_out_dir = out_dir)
