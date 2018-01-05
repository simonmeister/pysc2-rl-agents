# PySC2 Deep RL Agents

This repository attempts to implement the Advantage Actor-Critic (A2C) agent baseline for the 
[pysc2](https://github.com/deepmind/pysc2/) 
environment as described in the 
[DeepMind StarCraft II paper](https://deepmind.com/documents/110/sc2le.pdf).

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



## Related repositories
- We borrowed some code from [sc2aibot](https://github.com/pekaalto/sc2aibot)
- [pysc2-agents](https://github.com/xhujoy/pysc2-agents)
