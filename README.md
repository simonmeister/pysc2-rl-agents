# PySC2 Deep RL Agents

This repository attempts to implement a Advantage Actor-Critic agent baseline for the 
[pysc2](https://github.com/deepmind/pysc2/) 
environment as described in the 
[DeepMind StarCraft II paper](https://deepmind.com/documents/110/sc2le.pdf).
We use a synchronous variant of A3C (A2C) to effectively train on GPUs.

Note that this is still work in progress.

### License

This project is licensed under the MIT License (refer to the LICENSE file for details).

### Progress
- [x] A2C agent
- [x] FullyConv architecture
- [x] support all spatial screen and minimap observations
- [x] support the full action space as described in the DeepMind paper (predicting all arguments independently)
- [x] support training on all mini games
- [x] train MoveToBeacon
- [ ] train other mini games and correct any training issues
- [ ] support all non-spatial observations
- [ ] LSTM architecture
- [ ] Multi-GPU training

Any mini game can in principle be trained with the current code, although we still have to do experiments on maps other than `MoveToBeacon`.

## Results

| Map | mean score (ours) | mean score (DeepMind) |
| --- | --- | --- |
| MoveToBeacon | 25 | 26 |

With default settings (32 environments), learning MoveToBeacon currently takes between 3K and 10K episodes in total. This varies each run depending on random initialization and action sampling.

## Usage

### Hardware requirements
- for fast training, a GPU is recommended

### Software requirements
- Python 3
- pysc2 (tested with v1.2)
- TensorFlow (tested with 1.4.0)
- StarCraft II and mini games (see below or [pysc2](https://github.com/deepmind/pysc2/))

### Quick install guide
- `pip install numpy tensorflow-gpu pysc2`
- Install StarCraft II. On Linux, use [3.16.1](http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip).
- Download the [mini games](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

### Train & run
- train with `python run.py my_experiment --map MoveToBeacon`.
- run trained agents with `python run.py my_experiment --map MoveToBeacon --eval`.

You can visualize the agents with the `--vis` flag. 
See `run.py` for all arguments.

Summaries are written to `out/summary/<experiment_name>` and model checkpoints are written to `out/models/<experiment_name>`.


## Acknowledgments
The code in `rl/environment.py` is based on 
[OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/a2c),
with adaptions from
[sc2aibot](https://github.com/pekaalto/sc2aibot).
Some of the code in `rl/agents/a2c/runner.py` is loosely based on
[sc2aibot](https://github.com/pekaalto/sc2aibot).

Also see [pysc2-agents](https://github.com/xhujoy/pysc2-agents) for a similar repository.
