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
- [x] support all spatial screen and minimap observations
- [x] FullyConv architecture
- [x] support the full action space as described in the DeepMind paper (predicting all arguments independently)
- [x] train MoveToBeacon
- [ ] train other minigames
- [ ] support all non-spatial observations
- [ ] LSTM architecture
- [ ] Multi-GPU training

## Usage

### Hardware requirements
- for fast training, a GPU is recommended

### Software requirements
- Python 3
- pysc2 (tested with v1.2)
- TensorFlow (tested with 1.4.0)

### Run & evaluate experiments
- train with `python run.py my_experiment`.
- run trained agents with `python run.py my_experiment --eval`.

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
