<div align="center">
  <a href="https://youtu.be/m2pC9md0ixY" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123695-c69e17dc-104d-11e8-80bd-33726f3f0f12.gif"
         alt="MoveToBeacon"
         width="320" border="10" style="color:white" />
  </a>
  <a href="https://youtu.be/lpOlKfyhIXc" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123698-c910b650-104d-11e8-8019-8825187b677f.gif"
         alt="CollectMineralShards"
         width="320" border="10" style="color:white" />
  </a>
</div>

<div align="center">
  <a href="https://youtu.be/GFRsXx0imHc" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36123701-cabcb13e-104d-11e8-9aa8-7f1332d3cdb7.gif"
         alt="FindAndDefeatZerglings"
         width="320" border="10" style="color:white" />
  </a>
  <a href="https://youtu.be/-wDhAHkj90A" target="_blank">
    <img src="https://user-images.githubusercontent.com/22519290/36202298-87b14c60-1183-11e8-9b3f-f9bb5c8b1ab7.gif"
         alt="DefeatZerglingsAndBanelings"
         width="320" border="10" style="color:white" />
  </a>
</div>

# PySC2 Deep RL Agents

This repository implements a Advantage Actor-Critic agent baseline for the
[pysc2](https://github.com/deepmind/pysc2/)
environment as described in the DeepMind paper
[StarCraft II: A New Challenge for Reinforcement Learning](https://deepmind.com/documents/110/sc2le.pdf).
We use a synchronous variant of A3C (A2C) to effectively train on GPUs and
otherwise stay as close as possible to the agent described in the paper.

This repository is part of a research project at the
[Autonomous Systems Labs](http://www.ias.informatik.tu-darmstadt.de/)
, [TU Darmstadt](https://www.tu-darmstadt.de/) by
[Daniel Palenicek](https://github.com/danielpalen),
[Marcel Hussing](https://github.com/marcelhussing), and
[Simon Meister](https://github.com/simonmeister).

### Progress
- [x] A2C agent
- [x] FullyConv architecture
- [x] support all spatial screen and minimap observations as well as non-spatial player observations
- [x] support the full action space as described in the DeepMind paper
(predicting all arguments independently)
- [x] support training on all mini games
- [ ] report results for all mini games
- [ ] LSTM architecture
- [ ] Multi-GPU training

### License

This project is licensed under the MIT License (refer to the LICENSE file for details).

## Results

On the mini games, we get the following results:

| Map | best mean score (ours) |  best mean score (DeepMind) | episodes (ours) |
| --- | --- | --- | --- |
| MoveToBeacon | **26** | 26 | 8K |
| CollectMineralShards | **97** | 103 | 300K |
| FindAndDefeatZerglings | **45** | 45 | 450K |
| DefeatRoaches | **65** | 100 | - |
| DefeatZerglingsAndBanelings | **68** | 62 | - |
| CollectMineralsAndGas | - | 3978 | - |
| BuildMarines | - | 3 | - |

In the following we show plots for the score over episodes.

##### MoveToBeacon
<img src="https://user-images.githubusercontent.com/22519290/35693701-3b4b469a-077f-11e8-942d-c645644fa671.png" width="360">

##### CollectMineralShards
<img src="https://user-images.githubusercontent.com/22519290/35692346-66e966fa-077b-11e8-8ad3-03afc684b8bf.png" width="360">

##### FindAndDefeatZerglings
<img src="https://user-images.githubusercontent.com/22519290/35964384-e7ebe570-0cb7-11e8-90af-9ad495d42469.png" width="360">

Note that the DeepMind mean scores are their best individual scores after 100 runs for each
game, where the initial learning rate was randomly sampled for each run.
We use a constant initial learning rate for a much smaller number of runs due to limited hardware.
All agents use the same FullyConv agent.

With default settings (32 environments), learning MoveToBeacon well takes between 3K and 8K total episodes.
This varies each run depending on random initialization and action sampling.

## Usage

### Hardware Requirements
- for fast training, a GPU is recommended.
We ran each experiment on a single Titan X Pascal (12GB).

### Software Requirements
- Python 3
- pysc2 (tested with v1.2)
- TensorFlow (tested with 1.4.0)
- StarCraft II and mini games (see below or
  [pysc2](https://github.com/deepmind/pysc2/))

### Quick Install Guide
- `pip install numpy tensorflow-gpu pysc2==1.2`
- Install StarCraft II. On Linux, use
[3.16.1](http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip).
- Download the
[mini games](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

### Train & run
- run and train: `python run.py my_experiment --map MoveToBeacon`.
- run and evalutate without training: `python run.py my_experiment --map MoveToBeacon --eval`.

You can visualize the agents during training or evaluation with the `--vis` flag.
See `run.py` for all arguments.

Summaries are written to `out/summary/<experiment_name>`
and model checkpoints are written to `out/models/<experiment_name>`.


## Acknowledgments
The code in `rl/environment.py` is based on
[OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/a2c),
with adaptions from
[sc2aibot](https://github.com/pekaalto/sc2aibot).
Some of the code in `rl/agents/a2c/runner.py` is loosely based on
[sc2aibot](https://github.com/pekaalto/sc2aibot).
