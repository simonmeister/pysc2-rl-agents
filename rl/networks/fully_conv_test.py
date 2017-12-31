from collections import namedtuple

import tensorflow as tf
import numpy as np

from pysc2.lib.features import FeatureType, colors

from rl.networks.fully_conv import FullyConv


class FullyConvTest(tf.test.TestCase):
  def test_embed_obs(self):
    MyFeature = namedtuple('MyFeature', ['index', 'scale', 'type'])

    net = FullyConv()
    spec_lst = [
        (256, FeatureType.SCALAR),
        (4, FeatureType.CATEGORICAL),
        (2, FeatureType.SCALAR),
        (2, FeatureType.CATEGORICAL),
    ]
    spec = [MyFeature(i, *t) for i, t in enumerate(spec_lst)]
    embed_fn = lambda x, dims: x
    x = np.array([
        [112, 3, 0, 1],
        [22, 1, 1, 0],
        [0, 0, 0, 0],
        [255, 2, 1, 1]], dtype=np.float32)

    results = net.embed_obs(x, spec, embed_fn)
    expected_results = np.array([
        [1.5040774,    0, 0, 0, 1,    0,             0, 1],
        [0.52324814,   0, 1, 0, 0,    1.60943791,    1, 0],
        [0.0,          1, 0, 0, 0,    0,             1, 0],
        [2.19374631,   0, 0, 1, 0,    1.60943791,    0, 1]], dtype=np.float32)

    with self.test_session() as sess:
      results_out, = sess.run([results])
      self.assertAllClose(results_out, expected_results)
      

if __name__ == '__main__':
  tf.test.main()
