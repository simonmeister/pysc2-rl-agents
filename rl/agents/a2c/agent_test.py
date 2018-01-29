from collections import namedtuple

import tensorflow as tf
import numpy as np

from rl.agents.a2c.agent import A2CAgent


TestArgType = namedtuple('ArgType', ['name'])
arg_type = TestArgType('arg')
A = np.array


class A2CAgentTest(tf.test.TestCase):

  def test_compute_policy_log_probs(self):
    from rl.agents.a2c.agent import compute_policy_log_probs

    available_actions = A([[1, 0, 1],
                           [1, 0, 0],
                           [1, 1, 1]], dtype=np.float32)

    fn_pi = A([[0.2, 0.0, 0.8],
               [1.0, 0.0, 0.0],
               [0.2, 0.7, 0.1]], dtype=np.float32)

    fn_ids = A([2, 0, 1], dtype=np.int32)

    arg_pi = {arg_type: A([[0.8, 0.2],
                           [0.0, 1.0],
                           [0.5, 0.5]], dtype=np.float32)}

    arg_ids = {arg_type: A([0, 1, -1], dtype=np.int32)}

    log_probs = compute_policy_log_probs(
      available_actions, (fn_pi, arg_pi), (fn_ids, arg_ids)
    )

    expected_log_probs = np.log([0.8, 1.0, 0.7]) + A([np.log(0.8), np.log(1.0), 0])

    with self.test_session() as sess:
      log_probs_out = sess.run(log_probs)
      self.assertAllClose(log_probs_out, expected_log_probs)


  def test_compute_policy_entropy(self):
    from rl.agents.a2c.agent import compute_policy_entropy
    available_actions = A([[1, 0, 1],
                           [1, 0, 0],
                           [1, 1, 1]], dtype=np.float32)

    fn_pi = A([[0.2, 0.0, 0.8],
               [1.0, 0.0, 0.0],
               [0.2, 0.7, 0.1]], dtype=np.float32)

    fn_ids = A([2, 0, 1], dtype=np.int32)

    arg_pi = {arg_type: A([[0.8, 0.2],
                           [0.0, 1.0],
                           [0.5, 0.5]], dtype=np.float32)}

    arg_ids = {arg_type: A([0, 1, -1], dtype=np.int32)}

    entropy = compute_policy_entropy(
      available_actions, (fn_pi, arg_pi), (fn_ids, arg_ids)
    )

    expected_entropy = (0.50040245 + 0.80181855) / 3.0 + (0.50040245) / 2

    with self.test_session() as sess:
      entropy_out = sess.run(entropy)
      self.assertAllClose(entropy_out, expected_entropy)


if __name__ == '__main__':
  tf.test.main()
