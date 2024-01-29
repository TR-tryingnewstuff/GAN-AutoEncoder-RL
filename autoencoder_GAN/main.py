#%%
import pandas as pd
import numpy as np
import rllib 
import ray
from ray import tune
from ray import air
import yfinance as yf
import tensorflow as tf
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.pg.pg import PGConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.catalog import ModelCatalog
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from classes import AdverserialAutoEncoder

from neural_networks import EncoderModel, GeneratorModel, DiscriminatorModel
from ray.rllib.models.catalog import ModelCatalog
from torch.optim import NAdam, Adam, RMSprop

ModelCatalog.register_custom_model("encoder", EncoderModel)
ModelCatalog.register_custom_model("generator", GeneratorModel)
ModelCatalog.register_custom_model('discriminator', DiscriminatorModel)


tickers_df = yf.Tickers(['AAPL', 'GOOG', 'MSFT']).history(start='2010-01-01') 
tickers_df = tickers_df['Close']#.pct_change().dropna()
tickers_df

len_lags = 40
len_future = 10
true_obs = sliding_window_view(tickers_df, len_lags, axis=0).swapaxes(1, 2)[:-len_future]
true_returns = sliding_window_view(tickers_df, len_future, axis=0).swapaxes(1, 2)[len_lags:]

print(true_obs.shape, true_returns.shape)

ray.rllib.utils.check_env(AdverserialAutoEncoder({
                                                
                                                'true_obs': true_obs,
                                                'true_returns': true_returns,

                                                }))


#%%
#del my_new_ppo
def policy_mapping_fn(agent_id, episode, **kwargs):
    
    return agent_id


env_config={
        
        'true_obs': true_obs,
        'true_returns': true_returns,

        }


config = (
    PPOConfig()
    .environment(AdverserialAutoEncoder, env_config=env_config)
    .framework('tf')
    .rollouts(num_rollout_workers=14)
    .training(lr=0.00002)
    .multi_agent(policies={
                        
                        'encoder': PolicySpec(None, None, None, config={'model': {'custom_model': 'encoder'}}), 
                        'generator': PolicySpec(None, None, None, config={'model': {'custom_model': 'generator'}}),
                        'discriminator': PolicySpec(None, None, None, config={'model': {'custom_model': 'discriminator'}})},
                        policy_mapping_fn=policy_mapping_fn)
    
    
)


results = tune.Tuner(  

    "PPO",

    run_config=air.RunConfig(stop={"training_iteration": 50},
                             checkpoint_config=air.CheckpointConfig(1, checkpoint_at_end=True)),

    param_space=config.to_dict(),

).fit()

# %%

path = results._results[0].checkpoint.path
print(path) 

from ray.rllib.algorithms.algorithm import Algorithm

my_new_ppo = Algorithm.from_checkpoint(path)
my_new_ppo

#%%


env = AdverserialAutoEncoder(({
                            
                            'true_obs': true_obs,
                            'true_returns': true_returns,

                            }))

obs = env.reset()[0]

for i in range(3):
    
    action = {}
    for value, key in zip(obs.values(), obs.keys()):
        action[key] = my_new_ppo.compute_single_action(value, policy_id=key, explore=False)   

    #print(action)
    obs, reward, done, trunc, info = env.step(action)
    #print(obs)

print(env.discriminator_target)
print(action['discriminator'])
import matplotlib.pyplot as plt 
plt.plot(obs['discriminator'])


#%%
env.get_discriminator_sample({'generator': np.arange(len_future*3).reshape(len_future, 3)})

if env.discriminator_target:
    plt.plot(env.obs['discriminator'])
    
# %%
