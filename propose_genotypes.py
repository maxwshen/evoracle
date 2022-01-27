'''
  Input: Time series data of mutation frequencies (provided as a table: time by mutation, where each entry in the table is a frequency).

  This script proposes full-length genotypes based on mutation covariation over time.
'''

import os

import util

params = {
  'wt_symbol': '.',

  'change_threshold': 0.01,
  'majority_threshold': 0.5,
  'split_threshold': 0.01,

  'single_muts': [],
  'single_mut_positions': [],
  'group_to_len': {},
}

'''
  Support
'''
def parse_custom_hyperparams(custom_hyperparams):
  '''
    Parses a '+'-delimited string of hyperparameter tuples '{name}:{value}'
  '''
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
    'wt_symbol': parse_funcs['str'],

    'change_threshold': parse_funcs['float'],
    'majority_threshold': parse_funcs['float'],
    'split_threshold': parse_funcs['float'],
  }

  # Parse hyperparams
  global params
  terms = custom_hyperparams.split('+') if '+' in custom_hyperparams else [custom_hyperparams]
  for term in terms:
    [kw, args] = term.split(':')
    if kw in params:
      parse = term_to_parse[kw]
      params[kw] = parse(args)
      print(kw, parse(args))

  return


'''
  Build genotypes
'''
def form_gt(changers, majorities, groups):
  wt_symbol = params['wt_symbol']

  changer_group_to_syms = {group_name: symbols for symbols, group_name in map(util.parse_nt_pos, changers)}
  majorities_group_to_syms = {group_name: symbols for symbols, group_name in map(util.parse_nt_pos, majorities)}

  gt = ''
  for gidx in range(len(groups)):
    if gidx in changer_group_to_syms:
      gt += changer_group_to_syms[gidx]
    elif gidx in majorities_group_to_syms:
      gt += majorities_group_to_syms[gidx]
    else:
      group = groups[gidx]
      gt += wt_symbol * params['group_to_len'][gidx]
  return gt


def get_all_single_mutants():
  '''
  '''
  print(f'Adding single mutants ...')
  all_mutations = params['single_muts']
  gts = set()
  len_gt = len(params['single_mut_positions'])
  for mut in all_mutations:
    single_symbol, pos = util.parse_single_mut(mut)

    template = ['.'] * len_gt
    pos_idx = params['single_mut_positions'].index(pos)
    template[pos_idx] = single_symbol
    gt = ''.join(template)
    gts.add(gt)

  return gts


def default_subgroup(group, diffs, split_threshold):
  subgroups = []
  used = set()
  for idx in range(len(group)):
    mut = group[idx]
    if mut in used:
      continue

    curr_group = [mut]
    used.add(mut)

    for jdx in range(len(group)):
      j_mut = group[jdx]
      if j_mut not in used:
        if diffs[idx] - split_threshold <= diffs[jdx] <= diffs[idx] + split_threshold:
          curr_group.append(j_mut)
          used.add(j_mut)
    subgroups.append(curr_group)

  return subgroups


'''
  Parse Symbols and linkage group index
    Consider splitting into multiple columns?
'''
def setup(ntposs):
  '''
    Parse ntpos col, filling in global dict params
    
    single_mut_positions: List of single symbols
    single_muts: List of 'symbol pos' for single symbols
  '''
  group_to_len = {}
  for ntp in ntposs:
    symbols, group_name = util.parse_nt_pos(ntp)
    group_to_len[group_name] = len(symbols)
  params['group_to_len'] = group_to_len
  ns = sum(group_to_len.values())
  params['single_mut_positions'] = list(range(ns))

  sorted_group_nms = sorted(group_to_len.keys())
  group_to_cum_idx = {g: sum(group_to_len[gt] for gt in sorted_group_nms[:gi]) for gi, g in enumerate(sorted_group_nms)}

  muts = set()
  for ntp in ntposs:
    symbols, group_name = util.parse_nt_pos(ntp)
    for i, s in enumerate(symbols):
      if s != '.':
        # Single symbol position is sum of previous group lens and position of single symbol within current group
        pos = group_to_cum_idx[group_name] + i
        muts.add(f'{s} {pos}')
  params['single_muts'] = sorted(list(muts))

  print(f'Found {ns} unique mutation positions in {len(group_to_len)} groups ...')
  return


'''
  Main
'''
def get_default_genotypes(om_df, groups):
  '''
  '''
  change_threshold = params['change_threshold']
  majority_threshold = params['majority_threshold']
  split_threshold = params['split_threshold']

  nt_pos = om_df['Symbols and linkage group index']
  gts = set()

  time_cols = [col for col in om_df if col != 'Symbols and linkage group index']
  for idx in range(len(time_cols) - 1):
    t0, t1 = time_cols[idx], time_cols[idx + 1]

    diff = om_df[t1] - om_df[t0]

    uppers = list(nt_pos[diff > change_threshold])
    downers = list(nt_pos[diff < -1 * change_threshold])
    majorities = list(om_df[om_df[t1] >= majority_threshold]['Symbols and linkage group index'])

    up_diffs = list(diff[diff > change_threshold])
    down_diffs = list(diff[diff < -1 * change_threshold])
    covarying_groups = default_subgroup(uppers, up_diffs, split_threshold) + default_subgroup(downers, down_diffs, split_threshold)

    for gt in covarying_groups:
      gts.add(form_gt(gt, majorities, groups))

  single_mutants = get_all_single_mutants()
  for sm in single_mutants:
    gts.add(sm)

  gts.add(params['wt_symbol'] * len(params['single_mut_positions']))
  return sorted(list(gts))


def propose_genotypes(obs_marginals, out_fn, options = ''):
  '''
    Proposes genotypes.
  '''
  if options:
    print(f'Using custom hyperparameters: {options}')
    parse_custom_hyperparams(options)

  setup(obs_marginals['Symbols and linkage group index'])
  groups = util.parse_read_groups(obs_marginals)

  print(f'Proposing genotypes ...')
  gts = get_default_genotypes(obs_marginals, groups)

  out_dir = os.path.dirname(out_fn)
  util.ensure_dir_exists(out_dir)
  print(f'Writing {len(gts)} genotypes to {out_fn} ...')
  with open(out_fn, 'w') as f:
    for gt in gts:
      f.write(f'{gt}\n')

  print('Done.')
  return gts

