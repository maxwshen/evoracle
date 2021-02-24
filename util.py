'''
'''
import os


'''
  Parsing
'''
def parse_single_mut(mut):
  [symbol, pos] = mut.split()
  return symbol, int(pos)


def parse_nt_pos(ntp):
  [symbols, group_name] = ntp.split()
  return symbols, int(group_name)


def parse_read_groups(om_df):
  '''
    Parses df['Symbols and linkage group index'].

    Forms read_groups, a list of list of ints representing
    single-mutation positions.
  '''
  group_to_len = {}
  for ntp in om_df['Symbols and linkage group index']:
    symbols, group_name = parse_nt_pos(ntp)
    group_to_len[group_name] = len(symbols)

  read_groups = []
  base = 0
  for g, glen in sorted(group_to_len.items()):
    read_groups.append(list(range(base, base + glen)))
    base += glen

  return read_groups


'''
  Utility
'''
def ensure_dir_exists(directory):
  # Guarantees that input dir exists
  if not os.path.exists(directory):
    try:
      os.makedirs(directory)
    except OSError:
      if not os.path.isdir(directory):
        print(f'ERROR: Could not create {directory}. Create it yourself and try running Evoracle again.')
        raise
  return
