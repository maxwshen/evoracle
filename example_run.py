import pandas as pd
import evoracle
import example_dataloader as exdl

# Load data
obs_reads_df = exdl.load_obs_reads_df('example_data/cry1ac_illumina_100nt_obsreads.csv')
proposed_gts = exdl.load_proposed_gts('example_data/cry1ac_illumina_100nt_proposedgts.txt')
read_segments = exdl.load_read_segments('example_data/cry1ac_illumina_100nt_read_groups.txt')

out_dir = 'example_evoracle_out/'

out = evoracle.predict(
  obs_reads_df,
  read_segments,
  proposed_gts,
  out_dir,
)