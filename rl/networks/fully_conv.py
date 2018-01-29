import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.lib import actions
from pysc2.lib import features

from rl.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES


class FullyConv():
  """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

  Both, NHWC and NCHW data formats are supported for the network
  computations. Inputs and outputs are always in NHWC.
  """

  def __init__(self, data_format='NCHW'):
    self.data_format = data_format

  def embed_obs(self, x, spec, embed_fn):
    feats = tf.split(x, len(spec), -1)
    out_list = []
    for s in spec:
      f = feats[s.index]
      if s.type == features.FeatureType.CATEGORICAL:
        dims = np.round(np.log2(s.scale)).astype(np.int32).item()
        dims = max(dims, 1)
        indices = tf.one_hot(tf.to_int32(tf.squeeze(f, -1)), s.scale)
        out = embed_fn(indices, dims)
      elif s.type == features.FeatureType.SCALAR:
        out = self.log_transform(f, s.scale)
      else:
        raise NotImplementedError
      out_list.append(out)
    return tf.concat(out_list, -1)

  def log_transform(self, x, scale):
    return tf.log(x + 1.)

  def embed_spatial(self, x, dims):
    x = self.from_nhwc(x)
    out = layers.conv2d(
        x, dims,
        kernel_size=1,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format)
    return self.to_nhwc(out)

  def embed_flat(self, x, dims):
    return layers.fully_connected(
        x, dims,
        activation_fn=tf.nn.relu)

  def input_conv(self, x, name):
    conv1 = layers.conv2d(
        x, 16,
        kernel_size=5,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv1" % name)
    conv2 = layers.conv2d(
        conv1, 32,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv2" % name)
    return conv2

  def non_spatial_output(self, x, channels):
    logits = layers.fully_connected(x, channels, activation_fn=None)
    return tf.nn.softmax(logits)

  def spatial_output(self, x):
    logits = layers.conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None,
                           data_format=self.data_format)
    logits = layers.flatten(self.to_nhwc(logits))
    return tf.nn.softmax(logits)

  def concat2d(self, lst):
    if self.data_format == 'NCHW':
      return tf.concat(lst, axis=1)
    return tf.concat(lst, axis=3)

  def broadcast_along_channels(self, flat, size2d):
    if self.data_format == 'NCHW':
      return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                     tf.stack([1, 1, size2d[0], size2d[1]]))
    return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                   tf.stack([1, size2d[0], size2d[1], 1]))

  def to_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 2, 3, 1])
    return map2d

  def from_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 3, 1, 2])
    return map2d

  def build(self, screen_input, minimap_input, flat_input):
    size2d = tf.unstack(tf.shape(screen_input)[1:3])
    screen_emb = self.embed_obs(screen_input, features.SCREEN_FEATURES,
                                self.embed_spatial)
    minimap_emb = self.embed_obs(minimap_input, features.MINIMAP_FEATURES,
                                 self.embed_spatial)
    flat_emb = self.embed_obs(flat_input, FLAT_FEATURES, self.embed_flat)

    screen_out = self.input_conv(self.from_nhwc(screen_emb), 'screen')
    minimap_out = self.input_conv(self.from_nhwc(minimap_emb), 'minimap')

    broadcast_out = self.broadcast_along_channels(flat_emb, size2d)

    state_out = self.concat2d([screen_out, minimap_out, broadcast_out])

    flat_out = layers.flatten(self.to_nhwc(state_out))
    fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu)

    value = layers.fully_connected(fc, 1, activation_fn=None)
    value = tf.reshape(value, [-1])

    fn_out = self.non_spatial_output(fc, NUM_FUNCTIONS)

    args_out = dict()
    for arg_type in actions.TYPES:
      if is_spatial_action[arg_type]:
        arg_out = self.spatial_output(state_out)
      else:
        arg_out = self.non_spatial_output(fc, arg_type.sizes[0])
      args_out[arg_type] = arg_out

    policy = (fn_out, args_out)

    return policy, value
