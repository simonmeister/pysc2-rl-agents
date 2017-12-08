import numpy as np

from pysc2.lib import actions
from pysc2.lib import features


NUM_FUNCTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale


class Preprocessor():

  def __init__(self, obs_spec):
    self.screen_channels = self.input_channels(features.SCREEN_FEATURES)
    self.minimap_channels = self.input_channels(features.MINIMAP_FEATURES)
    self.flat_channels = (
        NUM_FUNCTIONS
        + NUM_PLAYERS
        + obs_spec['player'][0] - 1)

  def get_input_channels(self):
    """Get static channel dimensions of network inputs."""
    return self.screen_channels, self.minimap_channels, self.flat_channels

  def input_channels(self, spec):
    return sum(1 if l.type == features.FeatureType.SCALAR
               else l.scale for l in spec)

  # TODO vectorize for multiple environments
  def preprocess_obs(self, obs):
    """Compute screen, minimap and flat network inputs from raw observations.
    """
    # TODO for minigames, mask missing actions?
    available_one_hot = np.zeros(NUM_FUNCTIONS, dtype=np.float32)
    available_one_hot[obs['available_actions']] = 1

    player_id_one_hot = np.zeros(NUM_PLAYERS,
                                 dtype=np.float32)
    player_id_one_hot[obs['player'][0]] = 1
    player_numeric = np.asarray(obs['player'][1:], dtype=np.float32)

    screen = self.preprocess_spatial(obs['screen'], features.SCREEN_FEATURES)
    minimap = self.preprocess_spatial(obs['minimap'], features.MINIMAP_FEATURES)

    flat = np.concatenate([
        available_one_hot,
        player_id_one_hot,
        np.log(player_numeric)])

    return screen, minimap, flat

  def preprocess_spatial(self, spatial, spec):
    """Normalize numeric feature layers and convert categorical values to
    one-hot encodings.
    """
    height, width = spatial.shape[1:3]
    is_numeric = np.array([l.type == features.FeatureType.SCALAR for l in spec],
                          dtype=bool)
    is_categorical = np.logical_not(is_numeric)
    scale = np.array([l.scale for l in spec], dtype=np.float32)
    categorical_scale = scale[is_categorical]
    categorical_channels = np.sum(categorical_scale)

    numeric = spatial[is_numeric]
    numeric_scale = scale[is_numeric]
    numeric_out = np.log(numeric / numeric_scale)

    categorical_values = spatial[is_categorical]

    # TODO complete this
    v = categorical_values.ravel()
    indices = y * (channel * height * width) + x * (channel * )
    channel_offsets = np.concatenate([[0], np.cumsum(categorical_scale[:-1])])
    channel_offsets = np.tile(
        np.expand_dims(np.expand_dims(channel_offsets, 1), 2),
        [1, height, width])
    y = np.arange(height)
    x = np.arange(width)
    categorical_out = np.zeros(categorical_channels * height * width,
                               dtype=np.float32)
    categorical_out[indices] = 1
    categorical_out = np.reshape(categorical_out, [-1, height, width])

    out = np.reshape(
        np.concatenate([numeric_out, categorical_out]),
        [height, width, -1])
    return out
