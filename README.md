# Evoracle
A method for reconstructing frequency trajectories and fitnesses of long genotypes from short read data with missing linkage between polymorphic alleles from directed evolution timepoints.

Input: Time series data of mutation frequencies (provided as a table: time by mutation, where each entry in the table is a frequency). The user is expected to provide this data in a certain format, with guidelines and suggestions below.

Evoracle 1) proposes full-length genotypes that could exist in the population, then 2) infers the frequencies of full-length genotypes at each timepoint, and infers the fitness of each full-length genotype.

Output: A table (time by full-length genotype) of inferred frequencies. A table (full-length genotype by 1) of inferred fitness values.


## Workflow

A general workflow for using Evoracle with Illumina sequencing is:

Your job:
1. Perform DNA sequencing at multiple directed evolution timepoints. Convert bcl files to fastq. We recommend >10,000x coverage of your gene (e.g., for a 3000-nt gene with 150-nt reads, at least 200k reads).
2. Align fastq reads to your reference sequence. We recommend using the bowtie2 aligner.
3. Decide on a subset of nucleotide positions in your gene to retain for downstream analysis by domain knowledge or exploratory data analysis. While most nucleotides will have rare mutations at some timepoint, we recommend keeping the nucleotide positions that have a mutation with >5% (or >1% or >10%) population frequency at any timepoint. This typically preserves a few dozen nucleotide positions. (Note: Optionally, before this step, translate nucleotides into amino acids and call amino acid mutations.)
4. Calculate a table of mutation frequencies (called `obs_reads_df` here), where each column is a timepoint, each row is 1 or more mutations within a linkage group (i.e., nearby mutations that are jointly observed within a single short read), and each entry is the observed population frequency of reads containing that group of mutations. This is the expected input to Evoracle.

    Using the Evoracle code from this package:

5. Using Evoracle, propose full-length genotypes that may exist in the population using your table `obs_reads_df`. You can manually check and edit these full-length genotypes to incorporate domain knowledge.
6. Using Evoracle, run inference. Using your table `obs_reads_df`, a list of proposed full-length genotypes, and read groups, Evoracle infers a table of full-length genotype frequencies at each timepoint, and infers a fitness value for each full-length genotype.

A general workflow for using Evoracle with pooled Sanger sequencing is similar. Use Surveyor (https://www.bioke.com/webshop/sg/mutation-surveyor-softgenetics.html), EditR (https://www.liebertpub.com/doi/full/10.1089/crispr.2018.0014), Tracy (https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-6635-8) or similar tools to call nucleotide frequencies at each position at each timepoint using a reference sequence. Then proceed with step 3.


## Dependencies
The code was developed with Python 3.7 with `pandas==0.24.2` and `numpy==1.16.2`, `pytorch==1.5.0` and `torchvision==0.2.2`.


## Installation
Clone this github repository, then in Python, import the `evoracle.py` script. For instance, you may use the following at the top of your script to import the model.

```python
import sys
sys.path.append('/dir/to/local/repo/clone/')
import evoracle
```


## Usage
The simplest use of Evoracle combines steps 5 and 6 into one call: 
```python
import evoracle
evoracle.propose_gts_and_infer_fitness_and_frequencies(obs_reads_df,
  proposed_gt_out_fn, inference_out_dir)
```

Here, `obs_reads_df` is used as input. Evoracle proposes full-length genotypes and writes them to file at `proposed_gt_out_fn`. The output objects of inference, including a table of inferred genotype frequencies, and a vector of fitness values for each full-length genotype, and optimization logs, are written to `inference_out_dir`.

Alternatively, you can propose genotypes and run inference separately.

```python
import evoracle
gts = evoracle.propose_gts(obs_reads_df, out_fn)

evoracle.infer_fitness_and_frequencies(
  obs_reads_df,
  gts,    # Use gts from above, or load list from file
  out_dir)
```


### Running Evoracle with example data

We have included an example directed evolution dataset from Badran et al. 2015 (https://doi.org/10.1038/nature17938), with 34 timepoints collected every 12h or 24 h over a total of 528 h. The PacBio long-read sequencing dataset serves as a "ground truth" against which you can compare the model's inferred reconstructions. The model input are frequencies of 19 common non-synonymous mutations observed in 100-nt segments from 150-nt Illumina sequencing reads. Some 100-nt segments contain multiple mutations.

This code (also provided as `example_run.py`) runs Evoracle on the Badran dataset.

```python

# Load data
import pandas as pd
obs_reads_df = pd.read_csv(
  inp_dir + 'example_data/cry1ac_illumina_100nt_obsreads.csv'
)

# Run Evoracle
import evoracle
evoracle.propose_gts_and_infer_fitness_and_frequencies(
  obs_reads_df = obs_reads_df, 
  proposed_gt_out_fn = 'example_data/cry1ac_proposedgts.txt', 
  inference_out_dir = 'example_evoracle_out_cry1ac/')
```

All input and output files for this example run are provided in this github repo.


## Expected format of `obs_reads_df`
`obs_reads_df` is expected to be a pandas dataframe without an index column. There should be one column named 'Symbols and linkage group index' where each value is a string containing 1 or more symbols (alleles) and an integer separated by a space, for example `['.. 0, 'V. 0', '.I 0', 'VI 0', '. 1', 'W 1']`. Wild-type alleles are expected to be denoted by `.`. A symbol is any single unicode character, which supports nucleotides (A/C/G/T), amino acids, or other notation. Linkage groups represent sets of mutations that are jointly measured within a single short read. In the provided example, linkage group 0 contains two mutation positions, where the first position is either `.` or `V`, and the second position is either `.` or `I`. Since we jointly measure both positions in a short read, we can observe the frequencies of all combinations: `..`, `V.`, `.I`, and `VI`. The total number of linkage groups should be the total number of distinct mutation clusters that cannot be jointly observed in any single read. Linkage group indices are expected to start at 0 and increment by 1.

The remaining columns represent time as a series of increasing integers beginning with 0 where 1 represents the greatest common factor of all consecutive time intervals. For example, if timepoints were collected at hours 0, 12, 24, 48, 72, the columns should be named 0, 1, 2, 4, 6 where an increment of 1 represents 12 h.

In the table, each entry should be a frequency between 0 and 1, representing the observed frequency of symbols in a linkage group at a timepoint.

## Output format of Evoracle
Evoracle proposes full-length genotypes as concatenations of symbols across distinct linkage groups.

In general, we recommend the user to separately track the relationship between mutation position in Evoracle's ge

The output of inference is a dict `d`, where:
- `d['fitness']` is a pandas dataframe with columns 'Full-length genotype' and 'Fitness'. This is also saved to `_final_fitness.csv` in the output directory.
- `d['genotype_matrix']` is a pandas dataframe with a column 'Full-length genotype' and the same timepoint columns as the input `obs_reads` dataframe. Each value is an inferred frequency from 0 to 1. This is also saved to `_final_fitness.csv` in the output directory.

<!-- ### Terminology
A 'position' is a non-negative integer representing a unique amino acid or nucleotide position that has a mutation. Examples include `0`, `1`, etc.

A 'read segment' is a list of positions which can include one or more positions. In the case of the Cry1ac data, the first 100-nt read segment includes two positions with common mutations. Examples include `[0, 1]` and `[2]`.

A 'symbol' is a string representing the genotype of a read segment, and serves as an abstraction to encompass situations with both multiple mutations and single mutations. While any characters are supported, our formatting uses `.` to represent a single wild-type amino acid. Examples include `..`, `V.`, `.I`, and `VI`, which are the symbols in the first 100-nt read segment in the example Cry1ac data. Further examples include `.` and `W` from the second 100-nt read segment.

A 'full-length genotype' is a concatenation of symbols across all read segments. As an example, if there are two read segments where the first segment's symbols are only `.` and `A`, and the second's symbols are only `.` and `B`, then a full-length genotype is one of `['..', 'A.', '.B', 'AB']`. 


### Input data

`read_segments` is expected to be a list of read segments (see Terminology section). Example: `[[0, 1], [2], [3], [4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16], [17], [18]]`.

`proposed_gts` is expected to be a list of full-length genotypes (see Terminology section). Example: `['...................', 'VI..........YC.....', 'VIW...NGE.I.YC.KS.L']`.

 -->

## FAQ

__How should I represent variable time intervals?__

Times are expected to be represented as a list of integers starting from 0, where the interval 1 is equal to the greatest common factor among all time intervals. For a specific example, refer to the Cry1ac dataset which was sequenced every 12 h from hour 0 to 276, then every 24 h afterard until hour 528. This is represented by [0, 1, 2, ..., 20, 22, 24, ..., 46] where the interval 1 represents 12 h. Mathematically, if there are two timepoints X and X+T, the Evoracle model simulates T steps of natural selection. 

__How does Evoracle handle multiple mutations at the same position?__

Refer to the ABE8e dataset for an example. The last position has multiple possible symbols `N`, `G`, and `.` (wild-type).

__What advanced options are available for inference?__

For advanced use, the `evoracle.predict` function supports a parameter `option` which takes in a string, allowing control over hyperparameters such as training step size, num. iterations, weights on loss function terms. `option` parses string arguments as "(param_name):(value)+(param_name2):value2)+..." without parantheses. We refer the advanced user to read the code in the function `parse_custom_hyperparams` in `evoracle.py` for more details on writing string arguments for `option`.


## Contact
maxwshen at gmail.com


### License
tba
