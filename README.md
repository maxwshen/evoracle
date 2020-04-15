# evo-tft
Evolutionary trajectory and fitness triangulation

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
  obs_reads = df, 
  read_groups = rg, 
  proposed_gts = proposed_gts,
  hparams = hparams,
)
```

`df` is expected to be a pandas dataframe.

`out` is ...

### Running the model with example data

We have included an example directed evolution dataset from Badran et al. 2015 (https://doi.org/10.1038/nature17938), with 34 timepoints collected every 12h or 24 h over a total of 528 h. The PacBio long-read sequencing dataset serves as a "ground truth" against which you can compare the model's inferred reconstructions. The model input will be frequencies of 19 common non-synonymous mutations observed in 100-nt segments from 150-nt Illumina sequencing reads. Some 100-nt segments contain multiple mutations.

```python
import pandas as pd
import evo_tft

df = pd.read_csv('/directory/containing/local/repo/clone/example_data/cry1ac_illumina_100nt_obsreads.csv', index_col = 0)

out = evo_tft.predict(df)
```

## Contact
maxwshen at gmail.com

### License
tba
