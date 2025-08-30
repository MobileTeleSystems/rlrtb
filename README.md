# Collection of articles about Reinforcement Learning in Real-Time Bidding

**Real-Time Bidding (RTB)** optimization is a complex problem that requires instant decisions in the context of highly dynamic auctions, non-stationary competition and limited budgets. Traditional methods are often not flexible enough to adapt to a rapidly changing environment, which opens up opportunities for applying **Reinforcement Learning (RL)** methods.

This repository brings together key theoretical materials, algorithms and practical tools for diving into the problem of RL bid optimization in RTB. Here you find a structured selection of educational resources on the basics of RL, reviews of modern algorithms (from classic DQN to advanced SAC and TD3), research on bid landscape forecasting, as well as examples of implementing an auction simulator and RL-based agents. 

Processed sample of dataset and links to popular benchmarks such as iPinYou are provided for experiments. Using such examples of agents and environment, you can customize the experiment for your tasks, embedding the fulfillment of certain goals into the logic of agents, experiment with complex reward functions, tune agents and so on.

## Introduction

**RTB** is the process of automated buying and selling of advertisements in real time. RTB allows advertisers to compete for ad impressions based on target audiences and other factors, making it an attractive tool for effective advertising placement.

Optimizing an RTB bid is challenging because it requires making a decision in a short period of time (usually within tens of milliseconds) and taking into account many factors, such as user history, the context of the openRTB request, and competition among other advertisers. In such conditions, the use of traditional bid optimization algorithms may not be effective enough.

The use of RL methods in RTB bid optimization will allow you to increase the efficiency of your campaigns by assigning a bid close to the optimal one for each openRTB request, improve conversion rates and reduce advertising costs.

## Basics
| Title | Short description | Link |
| ------ | ------ | ------ |
| Reinforcement Learning An Introduction second edition | Comprehensive introduction | [[Link]](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) |
| CS234: Reinforcement Learning | Lectures from Stanford RL course | [[Link]](https://web.stanford.edu/class/cs234/modules.html)|
| CS224R: Deep Reinforcement Learning | Lectures from Stanford Deep RL course | [[Link]](https://cs224r.stanford.edu/) |
| UCL Course on RL | David Silver's RL course |[[Link]](https://www.davidsilver.uk/teaching/)|
| CS 285: Deep Reinforcement Learning | Deep RL cousre at UC Berkeley| [[Link]](https://rail.eecs.berkeley.edu/deeprlcourse/)|
| DeepMind Advanced DL and RL | DL and RL couse (slides and videos) | [[Link]](https://github.com/enggen/DeepMind-Advanced-Deep-Learning-and-Reinforcement-Learning) |
| Foundations of Deep RL | Pieter Abbeel lectures (videos) | [[Link]](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0) |
| Lilian Weng’s blog | Detailed description of algorithms | [[Link]](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) |
| Algorithms of Reinforcement Learning | Lecture notes | [[Link]](https://sites.ualberta.ca/~szepesva/rlbook.html) |
| Reinforcement Learning: A Comprehensive Overview | Overview of the field of (Deep) RL | [[Link]](https://arxiv.org/abs/2412.05265v2) |
| Bandit Algorithms | Complete tutorial on Multi-Armed Bandit (MAB) problem | [[Link]](https://tor-lattimore.com/downloads/book/book.pdf) |
| COMS E6998.001: Bandits and Reinforcement Learning | Alex Slivkins MAB course | [[Link]](https://alekhagarwal.net/bandits_and_rl/) |
| Introduction to Multi-Armed Bandits | MAB book | [[Link]](https://arxiv.org/abs/1904.07272)|
| Hugging Face Deep RL course | Deep RL course with theory and practice | [[Link]](https://huggingface.co/learn/deep-rl-course/unit0/introduction) |
| Practical_RL | Open course on RL in the wild | [[Link]](https://github.com/yandexdataschool/Practical_RL?tab=readme-ov-file) |
| CSE 599: Adaptive Machine Learning (Winter 2018) | Online and Adaptive Methods for Machine Learning | [[Link]](https://courses.cs.washington.edu/courses/cse599i/18wi/) |
| Real-Time Bidding with Side Information | MAB regret analysis in online advertising auctions | [[Link]](https://www.mit.edu/~jaillet/general/bidding-nips-17.pdf)|
| Regret Minimization for Reserve Prices in Second-Price Auctions | Analysis of regret minimization algorithm for reserve price optimization in a second-price auction | [[Link]](https://cesa-bianchi.di.unimi.it/Pubblicazioni/secondPrice.pdf) |
| Real-Time Bidding by Reinforcement Learning in Display Advertising | Studying the bid decision process as a reinforcement learning problem | [[Link]](https://www.arxiv.org/pdf/1701.02490) |
| Efficient Algorithms for Stochastic Repeated Second-price Auctions | MAB (Upper Confidence Bound, UCB), analysis of regret in a second-price auction. | [[Link]](https://hal.science/hal-02997579v2/document) |
| Multi-Armed Bandit with Budget Constraint and Variable Costs | MAB (UCB-based) for constrained budgets and variable costs | [[Link]](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/mab-adx.pdf) |
| Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting| RTB basics, bid landscape, bidding strategies, dynamic pricing | [[Link]](https://arxiv.org/pdf/1610.03013.pdf)|
| Offline and Online Optimization with Applications in Online Advertising | Optimal bidding and pacing strategies | [[Link]](https://escholarship.org/content/qt9pc6s4kk/qt9pc6s4kk.pdf?t=qxd2mj)|
| ROI-Constrained Bidding via Curriculum-Guided Bayesian Reinforcement Learning | A framework for adaptively managing constraint-target tradeoffs in non-stationary advertising markets | [[Link]](https://www.arxiv.org/abs/2206.05240) |
| Bidding Machine: Learning to Bid for Directly Optimizing Profits in Display Advertising | Description of the bidder's work | [[Link]](https://arxiv.org/pdf/1803.02194.pdf)|
| Online Causal Inference for Advertising in Real-Time Bidding Auctions | Online method for performing causal inference in RTB advertising | [[Link]](https://www.arxiv.org/abs/1908.08600) |
| Real-Time Bidding A New Frontier of Computational Advertising Research | Description of auction types, bidding strategies, pacing (slides) | [[Link]](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/rtb-tutorial-wsdm.pdf) |
| Real-Time Bid Optimization with Smooth Budget Delivery in Online Advertising | Description of pacing types | [[Link]](https://arxiv.org/pdf/1305.3011.pdf) |
| Optimal Real-Time Bidding for Display Advertising | Bidding strategies overview | [[Link]](https://wnzhang.net/papers/ortb-kdd.pdf) |

## Bid Landcape Forecasting
| Title | Short description | Link | The year of publication | 
| ------ | ------ | ------ | ------ |
| Functional Bid Landscape Forecasting for Display Advertising | Bid landscape forecasting: tree-based, node splitting, survival modeling (slides) | [[Link]](https://www.saying.ren/slides/functional-bid-lands.pdf) | 2016 |
| Deep Landscape Forecasting for Real-time Bidding Advertising | Forecasting the bid landscape (Deep Learning) without making any assumptions about the distribution of rates for successive price patterns. | [[Link]](https://arxiv.org/pdf/1905.03028.pdf) | 2019 |
| Scalable Bid Landscape Forecasting in Real-time Bidding | Forecasting the price landscape (censored regression) with some simplifications/assumptions | [[Link]](https://arxiv.org/pdf/2001.06587.pdf) | 2020 |
| Arbitrary Distribution Modeling with Censorship in Real-Time Bidding Advertising | Neighborhood Likelihood Loss in bidding landscape forecasting problem | [[Link]](https://www.arxiv.org/pdf/2110.13587) | 2021 |

## Theory of algorithms
![ALT](/images/rl_taxonomy.png "RL taxonomy")
More attention is paid to **model-free** RL algorithms, since they are preferable for solving the problem of bid optimization in RTB. The advantage of this approach is not only to adhere to the theoretically optimal bidding strategy, but also to avoid the expensive computational costs associated with simulating an extremely dynamic non-stationary environment. 

| Title | Algorithm | Link | The year of publication | 
| ------ | ------ | ------ | ------ |
| Technical note: Q-learning | Q-learning | [[Link]](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) | 1992 |
| Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | REINFORCE | [[Link]](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) | 1992 |
| On-line Q-learning Using Connectionist Systems | SARSA | [[Link]](https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf) | 1994 |
| Policy Gradient Methods for Reinforcement Learning with Function Approximation | Policy Gradient (PG) | [[Link]](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | 1999 |
| Actor-Critic Algorithms  | Actor-Critic (AC) | [[Link]](https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | 1999 |
| Playing Atari with Deep Reinforcement Learning | Deep Q-Network (DQN) | [[Link]](https://arxiv.org/pdf/1312.5602.pdf)| 2013 |
| Deterministic Policy Gradient Algorithms | Deteministic Policy Gradient (DPG) | [[Link]](https://proceedings.mlr.press/v32/silver14.pdf) | 2015|
| Continuous Control with Deep Reinforcement Learning | Deep Deterministic Policy Gradient (DDPG) | [[Link]](https://arxiv.org/pdf/1509.02971.pdf) | 2015 |
| Asynchronous Methods for Deep Reinforcement Learning | Asynchronous Advantage Actor-Critic (A3C), Advantage Actor-Critic (A2C)  | [[Link]](https://arxiv.org/pdf/1602.01783) | 2016 |
| Deep Reinforcement Learning with Double Q-learning | Double Q-learning (Double DQN) | [[Link]](https://arxiv.org/pdf/1509.06461.pdf)| 2016 |
| Dueling Network Architectures for Deep Reinforcement Learning | Dueling Network Architectures (Dueling DQN) | [[Link]](https://arxiv.org/pdf/1511.06581.pdf) | 2016 |
| Prioritized Experience Replay | Prioritized Experience Replay (PER) (DQN replay buffer improvement) | [[Link]](https://arxiv.org/pdf/1511.05952.pdf) | 2016 | 
| Asynchronous Methods for Deep Reinforcement Learning | N-step Q-learning (N-step DQN) | [[Link]](https://arxiv.org/pdf/1602.01783.pdf) | 2016 |
| Proximal Policy Optimization Algorithms | Proximal Policy Optimization (PPO) | [[Link]](https://arxiv.org/pdf/1707.06347) | 2017 |
| Hindsight Experience Replay | Hindsight Experience Replay (HER) (wrapper) | [[Link]](https://arxiv.org/pdf/1707.01495.pdf) | 2017 |
| Rainbow: Combining Improvements in Deep Reinforcement Learning | RAINBOW (Combination of DQN improvements) | [[Link]](https://arxiv.org/pdf/1710.02298.pdf) | 2017 | 
| A Distributional Perspective on Reinforcement Learning | C51 (Categorical DQN) | [[Link]](https://arxiv.org/pdf/1707.06887.pdf) | 2017 |
| Distributional Reinforcement Learning with Quantile Regression | Quantile Regression DQN (QR-DQN) | [[Link]](https://arxiv.org/pdf/1710.10044v1.pdf)| 2017 |
| Implicit Quantile Networks for Distributional Reinforcement Learning | Impcit Quantile Networks (distributional generalization of the DQN) | [[Link]](https://arxiv.org/pdf/1806.06923.pdf) | 2018 |
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor | Soft Actor-Critic (SAC) | [[Link]](https://arxiv.org/pdf/1801.01290.pdf) | 2018 |
| Noisy Nets For Exploration | Noisy Nets (NN) | [[Link]](https://arxiv.org/pdf/1706.10295.pdf) | 2018 |
| Addressing Function Approximation Error in Actor-Critic Methods | Twin Delayed Deep Deterministic policy gradient (TD3) | [[Link]](https://arxiv.org/pdf/1802.09477.pdf) | 2018 |
| Distributed Distributional Deterministic Policy Gradients | Distributed Distributional Deterministic Policy Gradients (D4PG) | [[Link]](https://arxiv.org/pdf/1804.08617.pdf) | 2018 |

## RL in RTB
**Model-free approach: getting rid of modeling complexity**

As already mentioned above, creating an accurate model of the RTB auction dynamics is an extremely difficult task. The auction environment depends on many factors: the behavior of other bidders, changing user preferences, auction algorithms, which are often opaque. Trying to build an explicit model of this environment, which is necessary for **model-based** RL methods, can be prohibitively difficult and probably ineffective due to inevitable simplifications and inaccuracies.

In contrast, **model-free** RL methods learn directly from experience interacting with the environment, without the need for explicit modeling. The agent simply observes the state of the environment, takes actions (places bids), receives rewards (e.g. clicks, conversions or profits) and adjusts its strategy based on these observations. This makes model-free methods much more practical and flexible for application in complex and dynamic environments such as RTB auctions.

**Off-policy learning: learning from past experience and exploring new strategies**

Also, to learn effectively in RTB auctions, we often need to use data collected in the past or data obtained by exploring different bidding strategies.
Off-policy RL methods are ideal for this task.

**On-policy** methods, such as SARSA or Policy Gradient, learn directly from the experience gained from the agent’s current strategy. This means that to explore new strategies and improve the current one, the agent needs to constantly interact with the environment, generating new experience using the current strategy. This can be slow and inefficient, especially if exploring new strategies results in a temporary decrease in performance.

**Off-policy** methods, on the other hand, allow the agent to learn from the experience gained by any strategy, including past strategies or even random actions. This is achieved by separating the strategy used to collect data (behavior policy) and the strategy we are trying to optimize (target policy). This approach gives us a number of advantages:

1) Efficient use of experience: we can use accumulated data (e.g. from past simulations or even real auctions) to train the agent, even if this data was obtained using other strategies.

2) Experience Replay: Off-policy methods often use an Experience Replay Buffer, where past transitions (state, action, reward, next state) are stored. The agent can revisit and learn from these past experiences many times, which significantly improves training efficiency and stability.

3) Exploring new strategies more safely: we can explore new, potentially risky strategies by collecting data that can then be used to learn a more conservative and stable target strategy.

Of course, the algorithms are not without nuances and not without shortcomings ([Off-Policy Deep Reinforcement Learning without Exploration](https://www.arxiv.org/pdf/1812.02900)), but they are still good for our task.

| Title | Algorithm | Link | The year of publication | 
| ------ | ------ | ------ | ------ |
| Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising | DQN | [[Link]](https://browse.arxiv.org/pdf/1802.08365.pdf) | 2018 |
| Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising | MADDPG | [[Link]](https://arxiv.org/pdf/1802.09756.pdf) | 2018 |
| Real-Time Bidding with Soft Actor-Critic Reinforcement Learning in Display Advertising | SAC | [[Link]](https://fruct.org/publications/volume-25/fruct25/files/Yak.pdf) | 2019 |
| A Dynamic Bidding Strategy Based on Model-Free Reinforcement Learning in Display Advertising | TD3 | [[Link]](https://ieeexplore.ieee.org/document/9258910)| 2020 |
| Bid Optimization using Maximum Entropy Reinforcement Learning | SAC | [[Link]](https://arxiv.org/pdf/2110.05032.pdf) | 2021 | 
| Dynamic pricing under competition using Reinforcement Learning | DQN, SAC | [[Link]](https://link.springer.com/article/10.1057/s41272-021-00285-3) | 2021 | 
| Multi-Objective Actor-Critics for Real-Time Bidding in Display Advertising | DQN, A2C, A3C | [[Link]](https://www.arxiv.org/pdf/2002.07408) | 2022 |
| Real-time Bidding Strategy in Display Advertising: An Empirical Analysis | DQN, TD3 | [[Link]](https://arxiv.org/pdf/2212.02222.pdf) | 2022 | 

## Datasets and benchmarks
| Title | Short description | Paper link | Download link | The year of publication | 
| ------ | ------ | ------ | ------ | ------ |
| Real-Time Bidding Benchmarking with iPinYou Dataset | The most popular dataset/benchmark. Advertising campaigns refer to products from 9 different categories over 10 days in 2013. Contains 64.5M bids, 19.5M impressions, 14.79K clicks. Full dataset size approx 5.6 Gb. | [[Link]](https://arxiv.org/pdf/1407.7073.pdf) | [[Link]](https://contest.ipinyou.com/) | 2014 |
| User Response Learning for Directly Optimizing Campaign Performance in Display Advertising | A huge dataset of RTB data on advertising campaigns that ran for 10 days in 2016. Contains 402 million impressions and 500 thousand clicks. Full dataset size approx 88.0 Gb. | [[Link]](https://apex.sjtu.edu.cn/public/files/papers/20160817/opt-ctr-bid.pdf) | [[Link]](https://apex.sjtu.edu.cn/datasets/7) | 2016 |

## Related repositories
| Title | Short description | Link |
| ------ | ------ | ------ |
| OpenAI Spinning Up | Educational resource to help anyone learn Deep RL | [[Link]](https://github.com/openai/spinningup) |
| Stable-Baselines3 | PyTorch version of Stable Baselines, reliable implementations of RL algorithms | [[Link]](https://github.com/DLR-RM/stable-baselines3) |
| Paper Collection of Real-Time Bidding | A collection of research and survey papers of RTB based display advertising techniques | [[Link]](https://github.com/wnzhang/rtb-papers?tab=readme-ov-file) |
| Deep RL with PyTorch | PyTorch implementation of various algorithms | [[Link]](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch) |
| CleanRL | Single-file implementation of Deep RL algorithms with research-friendly features | [[Link]](https://github.com/vwxyzjn/cleanrl) |
| CORL | Single-file implementations of SOTA offline and offline-to-online RL algorithms| [[Link]](https://github.com/tinkoff-ai/CORL) |
| EasyRL | PyTorch implementation of various algorithms | [[Link]](https://github.com/datawhalechina/easy-rl/tree/master) |

## Useful resources
| Title | Short description | Link |
| ------ | ------ | ------ |
| OpenAI Spinning Up | Educational resource to help anyone learn Deep RL | [[Link]](https://spinningup.openai.com/en/latest/) |
| Stable-Baselines3 | Stable-Baselines3 Docs - reliable RL implementations | [[Link]](https://stable-baselines3.readthedocs.io/en/master/) |
| Gym | Collection of reference environments (moved to Gymnasium) | [[Link]](https://www.gymlibrary.dev/) |
| Gymnasium | Collection of reference environments | [[Link]](https://gymnasium.farama.org/) |
| Third-Party Environments (Gym list) | Collection of third-party environments | [[Link]](https://www.gymlibrary.dev/environments/third_party_environments/) |
| Third-Party Environments (Gymnasium list) | Collection of third-party environments | [[Link]](https://gymnasium.farama.org/environments/third_party_environments/) |

## Authors
* Dmitrii Frolov

## License
The MIT License (MIT)
Copyright © 2025 MTS ADTECH, LLC. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
