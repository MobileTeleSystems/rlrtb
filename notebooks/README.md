# Description

To quickly dive into the problem of price optimization, you can use a simulation of the auction advertising process and experiment with ready-made agents. The environment is slightly simplified, but allows you to understand the essence of the issue and can serve as a platform for experiments and development of agents for RTB auctions using the selected methods. A description of the environment and agents can be found in the relevant sections.

Our environment will send openRTB requests to the agent received from the dataset. Requests in this case are taken from the processed sample of openRTB dataset iPinYou augmented with click probability. When receiving an openRTB request to display an advertisement, the agent will set a bid according to his bidding strategy.
After we send the last request, we can estimate at what rate the agent bought traffic and how many target events, in this case clicks, were purchased.

## Get started

To start the experiment you need to:
1) Make sure that all the necessary libraries are installed (list below).

2) Write down the paths to the folders with agents and environment in the exmaple jupyter notebook (/rl-rtb-papers/notebooks/example.ipynb).

3) Write the path to the test data sample in the environment config file (/rl-rtb-papers/scripts/gym-rtb/rtb_env/envs/config.cfg).

4) Configure the algorithm parameters, make changes to the agent's goals, etc.

## Required libraries

Auction simulation is implemented under the following development environment:

- python==3.12.7
- numpy>=2.1.3
- pandas>=2.2.3
- torch>=2.6.0
- scipy>=1.15.0
- matplotlib>=3.9.2
- gymnasium>=1.0.0
- pyarrow>=19.0.0

The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:
```python3
pip install -r /path/to/rl_papers/rl-rtb-papers/requirements.txt
```