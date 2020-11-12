# Evoracle
A method for reconstructing long genotypes from short read data with missing linkage between polymorphic alleles from directed evolution timepoints.

## Dependencies
The code was written with Python 3.7 with pandas==0.24.2 and numpy==1.16.2. The models were built with pytorch==1.4.0 and torchvision==0.2.2.

## Installation
Clone this github repository, then in Python, import the `evo_tft.py` script. For instance, you may use the following at the top of your script to import the model.

```python
import sys
sys.path.append('/directory/containing/local/repo/clone/')
import evo_tft
```

## Usage
```python
import evo_tft
out = evo_tft.predict(
  obs_reads = obs_reads_df, 
  read_segments = read_segments, 
  proposed_gts = proposed_gts,
  out_dir = out_dir,
  hparams = hparams,
)
```

All function parameters are mandatory except `hparams`.

### Terminology
A 'position' is a non-negative integer representing a unique amino acid or nucleotide position that has a mutation. Examples include `0`, `1`, etc.

A 'read segment' is a list of positions which can include one or more positions. In the case of the Cry1ac data, the first 100-nt read segment includes two positions with common mutations. Examples include `[0, 1]` and `[2]`.

A 'symbol' is a string representing the genotype of a read segment, and serves as an abstraction to encompass situations with both multiple mutations and single mutations. While any characters are supported, our formatting uses `.` to represent a single wild-type amino acid. Examples include `..`, `V.`, `.I`, and `VI`, which are the symbols in the first 100-nt read segment in the example Cry1ac data. Further examples include `.` and `W` from the second 100-nt read segment.

A 'full-length genotype' is a concatenation of symbols across all read segments. As an example, if there are two read segments where the first segment's symbols are only `.` and `A`, and the second's symbols are only `.` and `B`, then a full-length genotype is one of `['..', 'A.', '.B', 'AB']`. 

### Input data
`obs_reads` is expected to be a pandas dataframe without an index column. There should be one column named 'Symbol and read segment number' where each value is a string containing a symbol and an integer separated by a space, for example `['.. 0, 'V. 0', '.I 0', 'VI 0', '. 1', 'W 1']`. The remaining columns should be integers beginning with 0 where 1 represents the greatest common factor of all consecutive time intervals. For example, if the experimental timepoints are at hours 0, 12, 24, 48, and 60, the columns should be named 0, 1, 2, 4, 5 where 12 h is the greatest common factor. In the table, each value should be a frequency between 0 and 1, representing the frequency of observing the specified symbol in the specified read segment at the specified timepoint.

`read_segments` is expected to be a list of read segments (see Terminology section). Example: `[[0, 1], [2], [3], [4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16], [17], [18]]`.

`proposed_gts` is expected to be a list of full-length genotypes (see Terminology section). Example: `['...................', 'VI..........YC.....', 'VIW...NGE.I.YC.KS.L']`.

`out_dir` is expected to be a directory.

`out` is a dict, where:
- `out['fitness']` is a pandas dataframe with columns 'Full-length genotype' and 'Fitness'
- `out['genotype_matrix']` is a pandas dataframe with a column 'Full-length genotype' and the same timepoint columns as the input `obs_reads` dataframe. Each value is an inferred frequency from 0 to 1.

### Running the model with example data

We have included an example directed evolution dataset from Badran et al. 2015 (https://doi.org/10.1038/nature17938), with 34 timepoints collected every 12h or 24 h over a total of 528 h. The PacBio long-read sequencing dataset serves as a "ground truth" against which you can compare the model's inferred reconstructions. The model input will be frequencies of 19 common non-synonymous mutations observed in 100-nt segments from 150-nt Illumina sequencing reads. Some 100-nt segments contain multiple mutations.

```python
import pandas as pd
import evo_tft

# Load data
obs_reads_df = pd.read_csv('/directory/containing/local/repo/clone/example_data/cry1ac_illumina_100nt_obsreads.csv')

out = evo_tft.predict(obs_reads_df)
```

## Contact
maxwshen at gmail.com

### License
tba
