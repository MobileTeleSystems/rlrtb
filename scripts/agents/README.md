# Agents description

**Constant Bid Agent** - agent who places the bid according to the media plan (e.g bidprice from request in our case).

**Budget Pacing Ageng** - adaptive bidding agent with budget pacing strategy for optimal campaign spending.

**Random Bid Agent** - agent who places the bid according to following formula:

random.randint(self.min_cpm, self.base_bid), where min_cpm is 0.6 * base_bid from media plan.

**Random Participation Bid Agent** (stub agent) - agent who makes a bid equal to the media plan, but sometimes exits the auction or enters randomly.

**Epsilon Greedy Bid Agent** (stub agent, especially for non-stationary environment) - agent who uses an ε-greedy strategy to balance exploration and exploitation when setting bids in ad auctions. It combines a base bid, predicted click-through rate (pCTR) based scaling, and random exploration with probability ε.

**Thompson Sampling Bid Agent** (stub agent) - agent who makes a bid equal to the media plan, based on sampled and observed CTR using Thompson Sampling.
It is a Bayesian approach to the multi-armed bandit problem that effectively balances exploration and exploitation by choosing actions based on a probabilistic model of the expected reward.

**UCB Bid Agent** (stub agent) - agent who makes a bid equal to the media plan, scaled by expected CTR using Upper Confidence Bound (UCB). Agent takes into account the uncertainty in assessing this clickability and balances between using known profitable options and exploring new opportunities via UCB algorithm.

**Linear Bid Agent** - agent who places the bid according to following formula:

self.base_bid * min(state['click_prob'] / bid_coef, 1), where base_bid - bid price from media plan; state['click_prob'] - predicted click probability/score for given request.

bid_coef = sum(self.click_probs) / len(self.click_probs) - scaling factor associated with expected impression utility.

**Q-Learning Bid Agent** - agent who makes a bid equal to the media plan, scaled by expected CTR based on the estimation of a Q-function that predicts the expected total reward for each state-action pair.

**DQN Bid Agent** - agent that makes bids using Deep Q-Network (DQN) with some improvements. DQN, in particular, uses neural networks to approximate the Q-function, which allows it to work with high-dimensional state spaces typical of RTB auctions.
Double DQN addresses an overestimation bias of Q-learning by decoupling selection and evaluation of the bootstrap action. Prioritized Experience Replay improves data efficiency, by replaying more often transitions from which there is more to learn. The Dueling network architecture helps to generalize across actions by separately representing state values and action advantages. Noisy networks uses stochastic network layers for exploration instead of ε-greedy policy.

**DDPG Bid Agent** - agent who makes bids using Deep Deterministic Policy Gradient (DDPG). This algorithm copes well with the problems with a continuous action space, which can be relevant if we want to model bids as a continuous value.

**TD3 Bid Agent** - agent who makes bids using Twin Delayed Deep Deterministic Policy Gradient (TD3). This algorithm is an improvement on DDPG aimed at increasing stability and preventing overestimation of the Q-function.

**SAC Bid Agent** - agent who makes bids using Soft Actor-Critic (SAC). This algorithm is also suitable for problems with continuous actions and is characterized by high learning efficiency and stability. SAC maximizes not only the expected reward, but also the entropy of the strategy, which facilitates exploration and finding more robust solutions.

# Note
A good **reward function** is key.

**Implicit reward**: it's critical that the reward is well defined and reflects the real goals of the advertising campaign: for example, the reward could be related to clicks, conversions, ROI, or a combination of these metrics.

**Reward is too sparse**: if the reward is only given for clicks (as implicitly implied by self.total_clicks += observation.get('click', 0) if reward else 0), it may be too sparse. Deep RL algorithms may have difficulty learning if it receives a reward infrequently. Consider using a denser reward, such as an intermediate reward for winning an auction even if a click does not occur immediately.

**Negative reward for losing**: while losing does not result in a positive reward, an explicit negative reward for losing (especially for losing by bidding too low) may help the agent learn to avoid low bids.