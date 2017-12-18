import sys
import os
import shutil
import sys
import argparse
from functools import partial

import tensorflow as tf

from rl.agents.a2c.runner import A2CRunner
from rl.agents.a2c.agent import A2CAgent
from rl.networks.fully_conv import FullyConv
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv


parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('experiment_id', type=str,
                    help='identifier to store experiment results')
parser.add_argument('--train', action='store_false',
                    help='if false, episode scores are evaluated')
parser.add_argument('--ow', action='store_true',
                    help='overwrite existing experiments (if --train=True)')
parser.add_argument('--map_name', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--visualize', action='store_true',
                    help='render with pygame (implies --envs=1)')
parser.add_argument('--resolution', type=int, default=64,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=64,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=40,
                    help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount for future rewards')
parser.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--summary_iters', type=int, default=50,
                    help='record summary after this many iterations')
parser.add_argument('--save_iters', type=int, default=5000,
                    help='store checkpoint after this many iterations')
parser.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy penalty')
parser.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--save_dir', type=str, default='out/summary',
                    help='root directory for checkpoint storage')
parser.add_argument('--summary_dir', type=str, default='out/models',
                    help='root directory for summary storage')


args = parser.parse_args()
if args.visualize:
  args.envs = 1
# TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)


ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)


def _save_if_training(agent, summary_writer):
  if args.train:
    agent.save(ckpt_path)
    summary_writer.flush()
    sys.stdout.flush()


def main():
    if args.train and args.ow:
      shutil.rmtree(ckpt_path, ignore_errors=True)
      shutil.rmtree(summary_path, ignore_errors=True)

    size_px = (args.resolution, args.resolution)
    env_args = dict(
        map_name=args.map_name,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        screen_size_px=size_px,
        minimap_size_px=size_px,
        visualize=args.visualize)

    envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * args.envs)
    # envs = SingleEnv(make_sc2env(**env_args))

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(summary_path)

    agent = A2CAgent(
        sess=sess,
        loss_value_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.lr)

    runner = Runner(
        envs=envs,
        agent=agent,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch,
        do_training=args.training)

    static_shape_channels = runner.preproc.get_input_channels()
    agent.build(static_shape_channels, resolution=args.resolution)

    if os.path.exists(ckpt_path):
      agent.load(ckpt_path)
    else:
      agent.init()

    runner.reset()

    i = 0
    try:
      while True:
        write_summary = i % args.summary_iters == 0:

        if i % args.save_iters == 0:
          _save_if_training(agent, summary_writer)

        result = runner.run_batch(train_summary=write_summary)

        if write_summary:
          agent_step, summary = result
          summary_writer.add_summary(summary, global_step=agent_step)
          print('iter %d' % i)

        i += 1

        if 0 <= args.iters <= i:
          break

    except KeyboardInterrupt:
        pass

    _save_if_training(agent, summary_writer)

    envs.close()
    summary_writer.close()


if __name__ == "__main__":
    main()
