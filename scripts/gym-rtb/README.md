# RTB auction simulator
RTBEnv-v0 environment mimics the Ad Exchange by receiving openRTB requests so the agents can interact with it using API.

The auction simulator is currently adapted to processed openRTB requests from [iPinYou](https://arxiv.org/abs/1407.7073) dataset. Later, the simulator with this generally accepted dataset can be used to evaluate our bidding strategies and debugging the algorithmic part of agents.

[Gymnasium](https://gymnasium.farama.org/index.html) library is involved here, but the library The OpenAI [gym](https://www.gymlibrary.dev/index.html) can also be used after making minor changes.

## Description
The environment was created according to the official [instructions](https://www.gymlibrary.dev/content/environment_creation/) and has the following structure:
```
gym-rtb/
├─ rtb_env/
│  ├─ envs/
│  │  ├─ __init__.py
│  │  ├─ config.cfg
│  │  ├─ env.py
│  ├─ __init__.py
├─ README.md
├─ setup.py

```

## Data format

- **timestamp** (numpy.int64): format yyyyMMddHHmmssSSS
- **weekday** (numpy.int64): day of week 0-6
- **hour** (numpy.int64): hour of day
- **bidid** (str): bid identifier
- **minute** (numpy.int64): minute (from timestamp)
- **logtype** (numpy.int64): impression flag
- **ipinyouid** (str): internal user id set by iPinYou
- **useragent** (str): user's device, browser and opertaion system
- **region** (numpy.int64): region id
- **city** (numpy.int64): city id
- **adexchange** (numpy.int64): ad exchange id
- **IP** (str): user's IP adress
- **domain** (str): domain name
- **url** (str): URL of the hosting webpage of the ad slot
- **urlid** (numpy.float64): URL, masked by ad exchanges
- **slotid** (str): slot id
- **slotwidth** (numpy.int64): slot width
- **slotheight** (numpy.int64): slot height
- **slotvisibility** (numpy.int64): describes if the ad slot is above the fold ('FirstView') or not ('SecondView'), or unknown ('Na')
- **slotformat** (numpy.int64): possible values include 'Fixed' (fixed size and position), 'Pop' (the pop-up window), 'Background', 'Float', and 'Na' which presents unknown cases.
- **creative** (str): creative id
- **slotprice** (numpy.int64): bid floor
- **bidprice** (numpy.int64): bid price
- **payprice** (numpy.int64): market price
- **pCTR** (numpy.float64): predicted CTR
- **keypage** (str): key page URL
- **advertiser** (numpy.int64): advertiser id
- **user_os** (str): user's OS (from useragent field)
- **user_browser** (str): user's browser version (from useragent field)
- **usertag** (str): parsed into user_os and user_browser 
- **click** (numpy.int64): click or not click
- **nclick** (numpy.int64): number of clicks
- **nconversation** (numpy.int64): number of conversions



## Usage example
First you need to prepare the data and write the path to it in the **config.cfg** file, specify the auction type and select the desired metric for the environment response. 

In this example, the row of the iPinYou test dataset for AD campaign 1458 will be used. Get the initial state of the environment by calling the **.reset()** method:
```
import sys
import gymnasium as gym

path_to_env = 'path_to/rl-rtb-papers/scripts/gym-rtb'
sys.path.append(path_to_env)

import rtb_env


env = gym.make(
    id='RTBEnv-v0',
    disable_env_checker=True,
    num_agents=2
)

obs, reward, cost, done = env.reset()
```
**obs** - OpenRTB request content, augmented by predicted click probability (pCTR field). user_os and user_browser fields were obtained from the useragent field and minute from the timestamp field. Ideally it should be augmented with the probability of a deeper event.
```
obs: {
    'click': 0,
    'weekday': 5,
    'hour': 8,
    'bidid': '1a4ca48ddc07453e4c480c7414caa683',
    'timestamp': 20130614083809237,
    'logtype': 1,
    'ipinyouid': 'Vh5ACn273Ha4Jdn',
    'useragent': 'windows_ie',
    'IP': '218.4.56.*',
    'region': 80,
    'city': 85,
    'adexchange': 1,
    'domain': 'eSMJl65odN5vJMb4JKTI',
    'url': '13f42736e73118b639de0c8b73ca29cb',
    'urlid': nan,
    'slotid': 'mm_26632216_3300745_10762414',
    'slotwidth': 728,
    'slotheight': 90,
    'slotvisibility': 1,
    'slotformat': 1,
    'slotprice': 0,
    'creative': '48f2e9ba15708c0146bda5e1dd653caa',
    'bidprice': 300,
    'payprice': 11,
    'keypage': 'bebefa5efe83beee17a3d245e7c5085b',
    'advertiser': 1458,
    'usertag': '10063,10006,10115',
    'nclick': 0,
    'nconversation': 0,
    'pCTR': 0.1769,
    'user_os': 'windows',
    'user_browser': 'ie',
    'minute': 38
}
```
**reward** - reward from the environment for a completed action. In our case, the **action** is the bid price. Since we didn't make a bid, the value is 0.
```
reward: 0
```
**cost** - depending on the type of auction - this is either the winner's bid (action in case of winning; 1st price) or the real market price (2nd price). Commission were not introduced to simplify the implementation.
```
cost 0
```
**done** - since we are limited in budget, the agent will be forced to stop placing bids if the budget is no longer enough for even one bid from the contract, i.e. base bid. 

If we have enough budget for the bet – True, otherwise – False.
```
done: False
```

This way we get the opportunity to test our agent on real OpenRTB requests.


## Note
When adding **platform commission factor** (platform_fee in config.cfg) to cost it is **necessary to revise agent status update block (i.e cost > 0)**.