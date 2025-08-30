from gymnasium.envs.registration import register

register(
    id='RTBEnv-v0',
    entry_point='rtb_env.envs:RTBEnv'
)
