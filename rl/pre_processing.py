import numpy as np

from pysc2.lib import actions
from pysc2.lib import features


NUM_FUNCTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def stack_ndarray_dicts(lst, axis=0):
  """Concatenate ndarray values from list of dicts
  along new axis."""
  res = {}
  for k in lst[0].keys():
    res[k] = np.stack([d[k] for d in lst], axis=axis)
  return res


def log_transform(x, scale=None):
  if scale is not None:
    x /= scale
  return np.log(8 * x + 1) # TODO empirically select multiplier


class Preprocessor():
  """Compute network inputs from pysc2 observations.

  See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
  for the semantics of the available observations.
  """

  def __init__(self, obs_spec):
    self.screen_channels = self.input_channels(features.SCREEN_FEATURES)
    self.minimap_channels = self.input_channels(features.MINIMAP_FEATURES)
    self.flat_channels = (
        NUM_FUNCTIONS
        + NUM_PLAYERS
        + obs_spec['player'][0] - 1)
    self.available_actions_channels = NUM_FUNCTIONS

  def get_input_channels(self):
    """Get static channel dimensions of network inputs."""
    return {
        'screen': self.screen_channels,
        'minimap': self.minimap_channels,
        'flat': self.flat_channels,
        'available_actions': self.available_actions_channels}

  def input_channels(self, spec):
    return sum(1 if l.type == features.FeatureType.SCALAR
               else l.scale for l in spec)

  def preprocess_obs(self, obs_list):
    return stack_ndarray_dicts(
        [self._preprocess_obs(o.observation) for o in obs_list])

  def _preprocess_obs(self, obs):
    """Compute screen, minimap and flat network inputs from raw observations.
    """
    # TODO for minigames, mask missing actions?
    available_one_hot = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
    available_one_hot[obs['available_actions']] = 1

    player_id_one_hot = np.zeros(NUM_PLAYERS,
                                 dtype=np.float32)
    player_id_one_hot[obs['player'][0]] = 1
    player_numeric = np.asarray(obs['player'][1:], dtype=np.float32)

    screen = self._preprocess_spatial(obs['screen'], features.SCREEN_FEATURES)
    minimap = self._preprocess_spatial(obs['minimap'], features.MINIMAP_FEATURES)

    flat = np.concatenate([
        available_one_hot,
        player_id_one_hot,
        log_transform(player_numeric)])
        # TODO control groups, cargo, multi select, build queue

    return {
        'screen': screen,
        'minimap': minimap,
        'flat': flat,
        'available_actions': available_one_hot}

  # TODO vectorize this?
  def _preprocess_spatial(self, spatial, spec):
    """Normalize numeric feature layers and convert categorical values to
    one-hot encodings.
    """
    height, width = spatial.shape[1:3]
    # TODO maybe use l.index in case the pysc2 order changes in the future
    is_numeric = np.array([l.type == features.FeatureType.SCALAR for l in spec],
                          dtype=bool)
    is_categorical = np.logical_not(is_numeric)
    scale = np.array([l.scale for l in spec], dtype=np.float32)

    numeric = spatial[is_numeric].astype(np.float32)
    numeric_scale = scale[is_numeric]
    numeric = np.reshape(numeric, [height, width, -1])
    numeric_out = log_transform(numeric, numeric_scale)

    categorical = spatial[is_categorical]
    categorical_scale = scale[is_categorical]
    categorical_out = []
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    for i, depth in enumerate(categorical_scale):
      depth = int(depth)
      values = categorical[i, :, :]
      one_hot = np.zeros([height, width, depth], dtype=np.float32)
      one_hot[y, x, values] = 1
      categorical_out.append(one_hot)
    categorical_out = np.concatenate(categorical_out, axis=-1)

    out = np.concatenate([numeric_out, categorical_out], axis=-1)
    return out
