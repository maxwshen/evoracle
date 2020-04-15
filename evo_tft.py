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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

#
log_fn = None
fold_nm = ''

#
hparams = {
  'num_epochs': 1000,
  'learning_rate': 0.1,
  'weight_decay': 1e-5,

  # learning rate scheduler
  'plateau_patience': 10,
  'plateau_threshold': 1e-4,
  'plateau_factor': 0.1,

  # If the fraction of genotypes from the previous timepoint constitutes lower than this threshold in the current timepoint, skip calculating enrichments. Too many new genotypes -- can cause instability
  'dilution threshold': 0.3,

  # Predicted genotype frequencies are always >0, but below this threshold are treated as 0 and ignored for calculating enrichment
  'zero threshold': 1e-6,

  'random_seed': 0,

  'alpha_marginal': 50,
  'beta_skew': 0.01,
}

##
# Support
##
def parse_custom_hyperparams(custom_hyperparams):
  # Defaults
  if custom_hyperparams == '':
    return

  parse_funcs = {
    'slash-separated list': lambda arg: [int(s) for s in arg.split('/')],
    'binary': lambda arg: bool(int(arg)),
    'int': lambda arg: int(arg),
    'float': lambda arg: float(arg),
    'str': lambda arg: str(arg),
    'bool': lambda arg: bool(arg),
  }

  term_to_parse = {
    'num_epochs': parse_funcs['int'],
    'learning_rate': parse_funcs['float'],

    # learning rate schedulers
    'plateau_patience': parse_funcs['int'],
    'plateau_factor': parse_funcs['float'],
    'plateau_threshold': parse_funcs['float'],

    'dilution threshold': parse_funcs['float'],
    'zero threshold': parse_funcs['float'],

    'random_seed': parse_funcs['int'],

    'alpha_marginal': parse_funcs['float'],
    'beta_skew': parse_funcs['float'],

    'synthetic_noise': parse_funcs['float'],

    'dataset': parse_funcs['str'],
  }

  # Parse hyperparams
  global hparams
  terms = custom_hyperparams.split('+') if '+' in custom_hyperparams else [custom_hyperparams]
  for term in terms:
    [kw, args] = term.split(':')
    if kw in hparams:
      parse = term_to_parse[kw]
      hparams[kw] = parse(args)
      print(kw, parse(args))

  return


def copy_model_script():
  from shutil import copyfile
  copyfile(__file__, model_dir + f'{NAME}.py')


def check_num_models():
  import glob
  dirs = glob.glob(out_dir + 'model*')
  return len(dirs)


def print_and_log(text):
  with open(log_fn, 'a') as f:
    f.write(text + '\n')
  print(text)
  return


def create_model_dir():
  num_existing = check_num_models()
  global model_dir
  if fold_nm == '':
    run_id = str(num_existing + 1)
  else:
    run_id = fold_nm

  model_dir = out_dir + 'model_' + run_id + '/'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  subprocess.check_output(f'rm -rf {model_dir}*', shell = True)
  print('Saving model in ' + model_dir)

  global log_fn
  log_fn = out_dir + '_log_%s.out' % (run_id)
  with open(log_fn, 'w') as f:
    pass
  print_and_log('model dir: ' + model_dir)
  return


def set_random_seed():
  rs = hparams['random_seed']
  print(f'Using random seed: {rs}')
  np.random.seed(seed = rs)
  torch.manual_seed(rs)
  return


##
# Model
##
class MarginalFitnessModel(nn.Module):
  def __init__(self, package):
    super().__init__()
    self.fitness = torch.nn.Parameter(
      torch.ones(package['num_genotypes']) * -1 + torch.randn(package['num_genotypes']) * 0.01
    ).to(device)
    self.fq_mat = torch.nn.Parameter(
      torch.randn(
        package['num_timepoints'], package['num_genotypes']
      )
    ).to(device)
    self.num_genotypes = package['num_genotypes']
    self.num_timepoints = package['num_timepoints']
    self.len_alphabet = package['len_alphabet']
    self.genotypes_tensor = package['genotypes_tensor'].to(device)
    self.nt_to_idx = package['nt_to_idx']
    self.num_positions = package['num_positions']

  def forward(self):
    '''
      Forward pass.

      Output
      * log_pred_marginals: (n_t, n_pos, len_alphabet)
      * fitness_loss: (1)
  
      Intermediates
      * log_pred_p1s: (n_t - 1, n_gt)
      * d_nonzero_idxs: (n_t - 1, n_gt)

      * genotypes_tensor: (n_gt, n_pos * len_alphabet), binary

        Multiply with fq_mat (n_t, n_gt)
        -> curr_genotype_fqs = (n_t, n_gt, n_pos * len_alphabet), float

        Sum across n_gt. 
        -> pred_marginals: (n_t, n_pos, len_alphabet)
    '''

    '''
      Marginals
      
      genotypes_tensor: (n_gt, n_pos * len_alphabet), binary
      Multiply with fq_mat (n_t, n_gt)
      -> pred_marginals: (n_t, n_pos * len_alphabet)
    '''
    fqm = F.softmax(self.fq_mat, dim = 1)
    pred_marginals = torch.matmul(fqm, self.genotypes_tensor)
    pred_marginals /= self.num_positions
    # Ensure nothing is zero
    pred_marginals = pred_marginals + 1e-10
    log_pred_marginals = pred_marginals.log()

    # Fitness
    fitness_loss = torch.autograd.Variable(torch.zeros(1).to(device), requires_grad = True)
    for t in range(self.num_timepoints - 1):
      p0 = fqm[t]
      p1 = fqm[t + 1]

      nonzero_idxs = torch.BoolTensor(
        [bool(p0[idx] >= hparams['zero threshold']) for idx in range(self.num_genotypes)]
      )
      p0 = p0[nonzero_idxs]
      p1 = p1[nonzero_idxs]

      # Ignore case where too many new genotypes by normalizing to 1. This ensures that KL divergence cannot be negative. Deweight loss by 1 / sum
      np1 = p1 / p1.sum()
      weight = p1.sum()

      if p1.sum() < hparams['dilution threshold']:
        # Too many new genotypes, can cause instability in model
        continue

      present_fitness = torch.exp(self.fitness[nonzero_idxs])
      mean_pop_fitness = torch.dot(p0, present_fitness)
      delta_p = torch.div(present_fitness, mean_pop_fitness)
      pred_p1 = torch.mul(p0, delta_p)
      log_pred_p1 = torch.log(pred_p1)

      n_gt = np1.shape[0]
      log_pred_p1 = log_pred_p1.reshape(1, n_gt)
      np1 = np1.reshape(1, n_gt)
      loss = weight * F.kl_div(log_pred_p1, np1, reduction = 'batchmean')
      if loss > 100:
        print('WARNING: Fitness KL divergence might be infinite')
        import code; code.interact(local=dict(globals(), **locals()))

      fitness_loss = fitness_loss + loss
    fitness_loss = fitness_loss / (self.num_timepoints - 1)

    # Frequency sparsity loss
    fq_sparsity_loss = torch.autograd.Variable(torch.zeros(1).to(device), requires_grad = True)

    # Skew by time
    for t in range(self.num_timepoints):
      d = fqm[t]
      m = d.mean()
      s = d.std()

      # skew = torch.pow( (m - d) / s, 3).mean()
      # skew = torch.pow(d - m, 3).mean() / torch.pow(s, 3)
      unnorm_skew = torch.pow(d - m, 3).mean()

      # minimize loss to maximize positive skew
      fq_sparsity_loss = fq_sparsity_loss - unnorm_skew

    fq_sparsity_loss = fq_sparsity_loss / self.num_timepoints

    # L1 loss on fitness to prevent explosion and encourage exponential distribution (half of laplace distribution)
    fitness_exp = self.fitness.exp()
    l1_fitness_loss = torch.autograd.Variable(torch.zeros(1).to(device), requires_grad = True)
    l1_fitness_loss = l1_fitness_loss + fitness_exp.sum()

    skew_fitness_loss = torch.autograd.Variable(torch.zeros(1).to(device), requires_grad = True)
    # skew = torch.pow( (fitness_exp.mean() - fitness_exp) / fitness_exp.std(), 3).mean()
    skew = torch.pow(fitness_exp - fitness_exp.mean(), 3).mean() / torch.pow(fitness_exp.std(), 3)
    skew_fitness_loss = skew_fitness_loss - skew

    return log_pred_marginals, fitness_loss, fq_sparsity_loss, l1_fitness_loss, skew_fitness_loss


##
# Training
##
def train_model(model, optimizer, schedulers, dataset):
  since = time.time()
  model.train()

  marginal_loss_f = nn.KLDivLoss(reduction = 'batchmean')
  fitness_loss_f = nn.KLDivLoss(reduction = 'batchmean')

  obs_marginals = dataset.obs_marginals.to(device)

  num_epochs = hparams['num_epochs']
  epoch_loss = 0.0
  losses = []
  for epoch in range(num_epochs):
    print_and_log('-' * 10)
    print_and_log('Epoch %s/%s at %s' % (epoch, num_epochs - 1, datetime.datetime.now()))

    running_loss = 0.0
    with torch.set_grad_enabled(True):
      # One batch per epoch
      batch_loss = torch.autograd.Variable(torch.zeros(1).to(device), requires_grad = True)

      # Predict
      log_pred_marginals, fitness_loss, fq_sparsity_loss, l1_fitness_loss, skew_fitness_loss = model()

      '''
        Loss
        1. Soft constrain matrix of frequencies to match observed marginals (performed here)
        2. Loss when explaining/summarizing matrix by fitness values (performed inside model forward pass)
      '''
      marginal_loss = torch.autograd.Variable(torch.zeros(1).to(device), requires_grad = True)

      loss = marginal_loss_f(log_pred_marginals, obs_marginals)
      marginal_loss = marginal_loss + loss

      if marginal_loss > 100:
        print('WARNING: Marginal KL divergence might be infinite')
        import code; code.interact(local=dict(globals(), **locals()))

      '''
        Combine loss and backpropagate
      '''

      '''
        Vanilla loss. Works on z2 example with AA, AC, CA, and CC.
      '''
      # batch_loss = marginal_loss + fitness_loss

      # batch_loss = 5 * marginal_loss + fitness_loss

      # batch_loss = fitness_loss + 5 * marginal_loss + 0.01 * fq_sparsity_loss
      batch_loss = fitness_loss + hparams['alpha_marginal'] * marginal_loss + hparams['beta_skew'] * fq_sparsity_loss

      '''
        Experimental losses -- for HFNAP data
      '''
      # batch_loss = marginal_loss + fitness_loss + 0.1 * fq_sparsity_loss
      # batch_loss = 10 * marginal_loss + fitness_loss + 0.01 * fq_sparsity_loss + 0.0001 * l1_fitness_loss + 0.1 * skew_fitness_loss
      # batch_loss = marginal_loss

      running_loss = batch_loss
      batch_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      del batch_loss

    # Each epoch
    marginal_loss = float(marginal_loss.detach().numpy())
    fitness_loss = float(fitness_loss.detach().numpy())
    epoch_loss = float(running_loss.detach().numpy())
    fq_sparsity_loss = float(fq_sparsity_loss.detach().numpy())
    l1_fitness_loss = float(l1_fitness_loss.detach().numpy())
    skew_fitness_loss = float(skew_fitness_loss.detach().numpy())
    schedulers['plateau'].step(epoch_loss)
    losses.append(epoch_loss)

    print_and_log(f'Mar {marginal_loss:.3E} Fit {fitness_loss:.3E} Fq {fq_sparsity_loss:.3E} FSp {l1_fitness_loss:.3E} FSk {skew_fitness_loss:.3E}: {epoch_loss:.3E}')

    # Callback
    if epoch % 10 == 0:
      torch.save(model.state_dict(), model_dir + f'model_epoch_{epoch}_statedict.pt')
      pm_df = form_marginal_df(log_pred_marginals, dataset)
      pm_df.to_csv(model_dir + f'marginals_{epoch}.csv')

      fitness = list(model.parameters())[0].detach().exp().numpy()
      fitness_df = pd.DataFrame({
        'Genotype': dataset.genotypes,
        'Inferred fitness': fitness,
      })
      fitness_df.to_csv(model_dir + f'fitness_{epoch}.csv')

      gt_df = form_predicted_genotype_mat(model, dataset)
      gt_df.to_csv(model_dir + f'genotype_matrix_{epoch}.csv')

    # Early stop
    if epoch > 15:
      if losses[-15] == losses[-1]:
      # if losses[-15] > losses[-1]: # testing
        print_and_log('Detected convergence -- stopping')
        break

  time_elapsed = time.time() - since
  print_and_log('Training Completed in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

  fitness = list(model.parameters())[0].detach().exp().numpy()
  fq_mat = list(model.parameters())[1].detach().softmax(dim = 1).numpy()
  pred_marginals = log_pred_marginals.exp().detach().numpy()
  return fitness, fq_mat, pred_marginals


def form_predicted_genotype_mat(model, dataset):
  fqm = F.softmax(model.fq_mat, dim = 1).detach().numpy()
  return pd.DataFrame(fqm.T, index = dataset.genotypes)


def form_marginal_df(log_pred_marginals, dataset):
  pred_marginals = log_pred_marginals.exp().reshape(dataset.num_timepoints, dataset.num_positions * dataset.len_alphabet).detach().numpy()
  pm_df = pd.DataFrame(pred_marginals.T)
  
  nt_poss = []
  for pos in range(dataset.num_positions):
    for c in dataset.alphabet:
      nt_poss.append(f'{c}{pos}')

  ntp_col = 'Nucleotide and position'
  pm_df[ntp_col] = nt_poss
  pm_df = pm_df.sort_values(by = ntp_col).reset_index(drop = True)
  return pm_df


##
# Dataset
##
class MarginalDirectedEvolutionDataset(Dataset):
  def __init__(self, df, proposed_genotypes, read_groups, training = True):
    '''
      Expects df with columns 'Nucleotide and position', and ints starting from 0 for timepoints

      Batch = pair of adjacent timepoints
    '''
    self.read_groups = read_groups

    parsed = self.parse_df(df)
    self.df = df.set_index('Nucleotide and position')
    self.timepoints = list(self.df.columns)

    self.num_timepoints = parsed['Num. timepoints']
    self.num_positions = parsed['Num. positions']
    self.alphabet = parsed['Alphabet']
    self.len_alphabet = len(parsed['Alphabet'])
    self.num_marginals = parsed['Num. marginals']
    self.nt_to_idx = {s: idx for idx, s in enumerate(self.alphabet)}

    self.genotypes = proposed_genotypes
    self.num_genotypes = len(self.genotypes)
    self.genotypes_tensor = self.init_genotypes_tensor()

    # Provided to model class
    self.obs_marginals = self.init_obs_marginals()
    self.package = {
      'nt_to_idx': self.nt_to_idx,
      'num_timepoints': self.num_timepoints,
      'genotypes_tensor': self.genotypes_tensor,
      'num_genotypes': self.num_genotypes,
      'len_alphabet': self.len_alphabet,
      'num_positions': self.num_positions,
    }
    pass

  def __len__(self):
    '''
      Not used in this model
    '''
    return -1

  def __getitem__(self, idx):
    '''
      Not used in this model
    '''
    return {}

  #
  def parse_df(self, df):
    '''
      Expects df with columns 'Nucleotide and position' (e.g., AC 2), and ints starting from 0 for timepoints.

      Nucleotide can be any length, position also. They are separated by a space.

      Alphabet is built from observed marginals.
    '''
    ntposs = df['Nucleotide and position']
    rs = [col for col in df.columns if col != 'Nucleotide and position']

    # Remove alphabet characters that are 0 in all timepoints. This requires that proposed genotypes do not contain any of these character + position combinations.
    import copy
    dfs = copy.copy(df)
    dfs['Nucleotide'] = [s.split()[0] for s in ntposs]
    orig_alphabet_size = len(set(dfs['Nucleotide']))
    dfs['Position'] = [int(s.split()[1]) for s in ntposs]
    poss = set(dfs['Position'])
    dfs['Total count'] = df[rs].apply(sum, axis = 'columns')
    dfs = dfs[['Nucleotide', 'Total count']].groupby('Nucleotide')['Total count'].agg(sum)
    nts = [c for c, tot in zip(dfs.index, list(dfs)) if tot > 0]
    new_alphabet_size = len(nts)

    print(f'Reduced alphabet size from {orig_alphabet_size} to {new_alphabet_size}.')
    print(nts)
    df = df.set_index('Nucleotide and position')
    parsed = {
      'Num. timepoints': len(df.columns),
      'Num. marginals': len(df),
      'Alphabet': sorted(nts),
      'Num. positions': len(poss),
    }
    return parsed

  def init_obs_marginals(self):
    '''
      Init obs_marginals: (n_t, n_pos, len_alphabet)
    '''
    df = self.df
    marginals_tensor = torch.zeros(self.num_timepoints, self.num_positions, self.len_alphabet)
    for t_idx, t in enumerate(self.timepoints):
      for pos in range(self.num_positions):
        rows = [f'{nt} {pos}' for nt in self.alphabet]
        marginals_tensor[t_idx][pos] = torch.Tensor(df[t].loc[rows])
    marginals_tensor = marginals_tensor.reshape(self.num_timepoints, self.num_positions * self.len_alphabet)
    marginals_tensor /= self.num_positions
    return marginals_tensor

  def init_genotypes_tensor(self):
    '''
      * genotypes = (n_gt, n_pos, len_alphabet), binary

      Only keep proposed genotypes that are consistent with alphabet from observed marginals. 
    '''
    retained_gts = []
    for gt_idx, gt in enumerate(self.genotypes):
      inconsistent_gt = False
      for pos in range(self.num_positions):
        read_group = self.read_groups[pos]
        start_pos = read_group[0]
        end_pos = read_group[-1]
        kmer = gt[start_pos : end_pos + 1]
        if kmer not in self.nt_to_idx:
          inconsistent_gt = True
          break
      if not inconsistent_gt:
        retained_gts.append(gt)
    
    print(f'Reduced {self.num_genotypes} genotypes to {len(retained_gts)} consistent with alphabet.')
    self.genotypes = retained_gts
    self.num_genotypes = len(retained_gts)

    genotypes_tensor = torch.zeros(self.num_genotypes, self.num_positions, self.len_alphabet)
    for gt_idx, gt in enumerate(self.genotypes):
      for pos in range(self.num_positions):
        read_group = self.read_groups[pos]
        start_pos = read_group[0]
        end_pos = read_group[-1]
        kmer = gt[start_pos : end_pos + 1]
        nt_idx = self.nt_to_idx[kmer]
        genotypes_tensor[gt_idx][pos][nt_idx] = 1
    genotypes_tensor = genotypes_tensor.reshape(self.num_genotypes, self.num_positions * self.len_alphabet)
    return genotypes_tensor


##
# Use
##
def load_data(dataset_nm):
  data_place = '/ahg/regevdata/projects/CRISPR-libraries/prj2/evolution/abe8e/out/'
  data_dir = f'{data_place}/data_multi/'

  df = pd.read_csv(data_dir + f'obs_reads_pivot_{dataset_nm}.csv')

  with open(data_dir + f'propose_genotypes_{dataset_nm}.txt') as f:
    lines = f.readlines()
  proposed_genotypes = [s.strip() for s in lines]

  read_groups = pickle.load(open(data_dir + f'read_groups_{dataset_nm}.pkl', 'rb'))

  data_package = {
    'df': df,
    'proposed_genotypes': proposed_genotypes,
    'read_groups': read_groups,
  }

  return data_package


def run_inference(modelexp_nm = '', exp_nm = '', custom_hyperparams = ''):
  print(custom_hyperparams)
  parse_custom_hyperparams(custom_hyperparams)
  set_random_seed()

  dataset_nm = hparams['dataset']
  data_package = load_data(dataset_nm)

  dataset = MarginalDirectedEvolutionDataset(
    data_package['df'], 
    data_package['proposed_genotypes'], 
    data_package['read_groups'],
  )
  updated_proposed_genotypes = dataset.genotypes

  print('Setting up...')
  model = MarginalFitnessModel(
    dataset.package
  ).to(device)

  for param in model.parameters():
    print(type(param.data), param.shape)

  optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = hparams['learning_rate'],
    weight_decay = hparams['weight_decay'],
  )

  schedulers = {
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      patience = hparams['plateau_patience'],
      threshold = hparams['plateau_threshold'],
      factor = hparams['plateau_factor'],
      verbose = True,
      threshold_mode = 'rel',
    )
  }

  global out_dir
  if modelexp_nm == '': modelexp_nm = 'unnamed'
  out_dir = out_dir + modelexp_nm + '/' if modelexp_nm != '' else out_dir
  util.ensure_dir_exists(out_dir)

  global fold_nm
  fold_nm = exp_nm
  create_model_dir()
  copy_model_script()

  fitness, fq_mat, pred_marginals = train_model(model, optimizer, schedulers, dataset)

  package = {
    'fitness': fitness,
    'fq_mat': fq_mat,
    'pred_marginals': pred_marginals,
    'updated_proposed_gts': updated_proposed_genotypes,
    'num_updated_proposed_gts': len(updated_proposed_genotypes),
    'fitness_df': pd.DataFrame({
      'Genotype': list(updated_proposed_genotypes), 
      'Inferred fitness': fitness,
    }),
    'genotype_matrix': pd.DataFrame(
      fq_mat.T, 
      index = updated_proposed_genotypes,
    ),
  }

  package['fitness_df'].to_csv(model_dir + f'_final_fitness.csv')
  package['genotype_matrix'].to_csv(model_dir + f'_final_genotype_matrix.csv')

  with open(model_dir + f'_final_package.pkl', 'wb') as f:
    pickle.dump(package, f)

  return package


'''
  Testing
'''
def run_test():
  '''
  '''
  package = run_inference(
    custom_hyperparams = 'dataset:simple--pace_num-2--threshold-5--read_len-1--min_gt_frequency-0.01--proposal_type-smart',
  )
  return


