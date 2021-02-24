'''
  Simple wrapper script for proposing genotypes and running inference
'''
import inference as inf
import propose_genotypes as pgt


def propose_gts(obs_reads_df, out_fn):
  '''
    Uses obs_reads_df, which describes observed frequencies of mutations (symbols) in distinct linkage groups, to propose full-length genotypes that could occur in the real population.
  '''
  return pgt.propose_genotypes(obs_reads_df, out_fn)


def infer_fitness_and_frequencies(obs_reads_df, proposed_gts, out_dir, options = ''):
  '''
    Uses obs_reads_df, which describes observed frequencies of mutations (symbols) in distinct linkage groups, and proposed_gts, a list of full-length genotypes that may occur in the population.

    Infers the frequency of each full-length genotype at each timepoint, and a fitness value for each full-length genotype. Saves output objects to out_dir, and returns them in a dict.
  '''
  return inf.predict(obs_reads_df, proposed_gts, out_dir, options = options)


def propose_gts_and_infer_fitness_and_frequencies(obs_reads_df, proposed_gt_out_fn, inference_out_dir, inference_options = ''):
  '''
    A combination of the above functions.

    From obs_reads_df, which describes observed frequencies of mutations (symbols) in distinct linkage groups.

    1. First propose full-length genotypes that could exist in the population
    2. Then, infer the frequency of each full-length genotype at each timepoint, and a fitness value for each full-length genotype. Saves output objects to out_dir, and returns them in a dict. 
  '''
  gts = propose_gts(obs_reads_df, proposed_gt_out_fn)
  return infer_fitness_and_frequencies(obs_reads_df, gts, inference_out_dir, options = inference_options)
