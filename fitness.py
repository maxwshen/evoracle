# 
import sys, string, pickle, subprocess, os, datetime, gzip, time
sys.path.append('/home/unix/maxwshen/')
import numpy as np, pandas as pd
import scipy
from collections import defaultdict
from mylib import util
import _config

import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
import torch.nn as nn
import glob

np.random.seed(seed = 0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

random_seed = 0
torch.manual_seed(random_seed)

#
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

#
hyperparameters = {
  'num_epochs': 100000,
  'learning_rate': 1,

  # learning rate scheduler
  'plateau_patience': 10,
  'plateau_threshold': 1e-4,
  'plateau_factor': 0.1,

  # If the fraction of genotypes from the previous timepoint constitutes lower than this threshold in the current timepoint, skip calculating enrichments. Too many new genotypes -- can cause instability
  'dilution threshold': 0.3,
}

##
# Model
##
class FitnessModel(nn.Module):
  def __init__(self, num_genotypes):
    super().__init__()
    self.fitness = torch.nn.Parameter(
      torch.randn(num_genotypes)
    )

  def forward(self, p0, nonzero_idxs, t_idx):
    '''
      Forward pass
    '''

    # Hard coded: Using 12 h = one generation, recognize time indices 24 h apart
    time_step = 1
    if int(t_idx) >= 22:
      time_step = 2

    input_ps = p0
    present_fitness = torch.exp(self.fitness[nonzero_idxs])
    for ts_idx in range(time_step):
      mean_pop_fitness = torch.dot(input_ps, present_fitness)
      delta_p = torch.div(present_fitness, mean_pop_fitness)
      pred_p1 = torch.mul(input_ps, delta_p)
      input_ps = pred_p1

    log_pred_p1 = torch.log(pred_p1)
    return log_pred_p1

##
# Training
##
def train_model(model, optimizer, schedulers, dataset):
  since = time.time()
  model.train()

  loss_func = nn.KLDivLoss(reduction = 'batchmean')

  num_epochs = hyperparameters['num_epochs']
  epoch_loss = 0.0
  losses = []
  for epoch in range(num_epochs):
    print('-' * 10)
    print('Epoch %s/%s at %s' % (epoch, num_epochs - 1, datetime.datetime.now()))

    running_loss = 0.0
    with torch.set_grad_enabled(True):
      # One batch per epoch
      batch_loss = torch.autograd.Variable(
        torch.zeros(1).to(device), 
        requires_grad = True
      )
      for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        (p0, p1) = sample['frequencies']
        nonzero_idxs = sample['nonzero_idxs']

        nonzero_idxs = nonzero_idxs.to(device)
        p0 = p0[nonzero_idxs].to(device)
        p1 = p1[nonzero_idxs].to(device)
        if len(p0) == 0:
          continue

        # Ignore case where too many new genotypes by normalizing to 1. This ensures that KL divergence cannot be negative. 
        np1 = p1 / p1.sum()
        np1 = np1.to(device)
        # batch_weight = p1.sum()
        batch_weight = 1

        if p1.sum() < hyperparameters['dilution threshold']:
          # Too many new genotypes, can cause instability in model
          continue

        log_pred_p1 = model(p0, nonzero_idxs, t_idx = sample_idx)
        # print(p0.sum(), p1.sum(), pred_p1.exp().sum())

        n_gt = np1.shape[0]
        log_pred_p1 = log_pred_p1.reshape(1, n_gt)
        np1 = np1.reshape(1, n_gt)
        loss = loss_func(log_pred_p1, np1)

        if loss > 10:
          import code; code.interact(local=dict(globals(), **locals()))

        batch_loss = batch_loss + loss * batch_weight
        running_loss += loss.item() * batch_weight

      # Step once per batch during training
      batch_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      del batch_loss

    # Each epoch
    epoch_loss = running_loss / len(dataset)
    schedulers['plateau'].step(epoch_loss)
    losses.append(epoch_loss)

    print('Loss: {:.3E}'.format(epoch_loss))

    # Early stop
    if epoch > 15:
      if losses[-15] == losses[-1]:
        print('Detected convergence -- stopping')
        break

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

  fitness = np.exp(list(model.parameters())[0].detach().numpy())
  return fitness


##
# Dataset
##
class DirectedEvolutionDataset(Dataset):
  def __init__(self, df, training = True):
    '''
      Expects df with columns 'Genotype', and ints starting from 0 for timepoints

      Batch = pair of adjacent timepoints
    '''
    self.genotypes = df['Genotype']
    self.num_genotypes = len(df['Genotype'])

    rs = [col for col in df.columns if col != 'Genotype']
    self.num_timepoints = len(rs)
    self.data = {col: torch.Tensor(list(df[col])) for col in rs}

    results = self.init_nonzero_idxs()
    self.nonzero_idxs = results['nonzero_idxs']
    self.all_ok_idxs = results['all_nonzero_idxs']


  def __len__(self):
    # Return num. batches, which are pairs of adjacent timepoints
    return self.num_timepoints - 1

  def __getitem__(self, idx):
    '''
      Batch = pair of adjacent timepoints.
      All returned items are torch tensors
    '''
    return {
      'frequencies': (self.data[str(idx)], self.data[str(idx + 1)]),
      'nonzero_idxs': self.nonzero_idxs[idx],
    }

  #
  def init_nonzero_idxs(self):
    dd = dict()
    all_nonzero_idxs = None
    results = dict()
    for t in range(self.num_timepoints - 1):
      p0 = self.data[str(t)]
      nonzero_idxs = [bool(p0[idx] != 0) for idx in range(self.num_genotypes)]
      dd[t] = torch.BoolTensor(nonzero_idxs)
      if all_nonzero_idxs is None:
        all_nonzero_idxs = nonzero_idxs
      else:
        all_nonzero_idxs = [bool(s or t) for s, t in zip(all_nonzero_idxs, nonzero_idxs)]

    return {
      'nonzero_idxs': dd,
      'all_nonzero_idxs': all_nonzero_idxs,
    }

##
# Use
##
def infer_fitness(df):
  dataset = DirectedEvolutionDataset(df)

  print('Setting up...')
  model = FitnessModel(dataset.num_genotypes).to(device)
  for param in model.parameters():
    print(type(param.data), param.shape)

  optimizer = torch.optim.SGD(
    model.parameters(), 
    lr = hyperparameters['learning_rate'],
    momentum = 0.5,
  )

  schedulers = {
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      patience = hyperparameters['plateau_patience'],
      threshold = hyperparameters['plateau_threshold'],
      factor = hyperparameters['plateau_factor'],
      verbose = True,
      threshold_mode = 'rel',
    )
  }

  fitness = train_model(model, optimizer, schedulers, dataset)

  # mask entries with no entries in adjacent timepoints
  masked_fitness = [f if ok_flag else np.nan for f, ok_flag in zip(fitness, dataset.all_ok_idxs)]

  return masked_fitness


'''
  Testing
'''
def test():
  test_fold = '/ahg/regevdata/projects/CRISPR-libraries/prj2/evolution/badran/out/pb_e_form_dataset/'
  inp_fn = test_fold + 'badran_pacbio_pivot_1pct.csv'

  # Run test
  print('Loading data...')
  df = pd.read_csv(inp_fn)

  # Munge
  df = df.rename(columns = {'Abbrev genotype': 'Genotype'})
  time_cols = [col for col in df.columns if col != 'Genotype']
  col_to_idx = {col: str(idx) for idx, col in enumerate(time_cols)}
  df = df.rename(columns = col_to_idx)

  print('Creating dataset...')

  fitness = infer_fitness(df)

  # for gt, fit in zip(df['Genotype'], fitness):
  #   print(f'{gt}, inferred: {fit:.2f}')

  fitness_df = pd.DataFrame({
    'Genotype': df['Genotype'],
    'Fitness': fitness
  })
  fitness_df = fitness_df.sort_values(by = 'Fitness', ascending = False).reset_index(drop = True)
  fitness_df.to_csv(out_dir + f'fullgt_fitness.csv')

  print(fitness_df.head())

  return


if __name__ == '__main__':
  test()