# PySC2 Deep RL Agents

This repository attempts to implement the Advantage Actor-Critic agent baseline for the 
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
For example, `python run.py my_test_model --envs 1`.

## Related repositories
- We borrowed some code from [sc2aibot](https://github.com/pekaalto/sc2aibot)
- [pysc2-agents](https://github.com/xhujoy/pysc2-agents)
