#%%
import pandas as pd
import numpy as np
import gymnasium
from gymnasium.spaces import Dict, Box
from ray.rllib.env.env_context import EnvContext 
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, mutual_info_score, explained_variance_score, mean_absolute_error
import copy
import matplotlib.pyplot as plt
import dtw

def cross_entropy(yreal, ypred):
    with np.errstate(divide='ignore', invalid='ignore'):
        
        if ypred == 1:
            ypred = 0.9999999999999
        elif ypred == 0:
            ypred = 0.0000000000001
        
        cross_entropy = (yreal*np.log(ypred) + (1-yreal)*np.log(1-ypred))

        return cross_entropy 

def dtw_distance(yreal, ypred):
    
    return dtw.dtw(yreal, ypred).distance

#%%
class AdverserialAutoEncoder(MultiAgentEnv):
    def __init__(self, config: EnvContext):
        super().__init__()

        self.true_obs = config['true_obs'] # should take the form of a numpy array, with lags
        self.true_returns = config['true_returns']
        self.length_train = 3
        self.length_eval = 3
        self.history = []
        
        self._agent_ids = ['encoder', 'generator', 'discriminator']
        
        self.truncated = {'__all__': False}
        self.Done = {'__all__': False}

        self.bins = 3
        encoder_output = np.zeros(self.bins*self.true_obs[0].shape[1]).shape
        lags = self.true_obs[0].shape
        future = self.true_returns[0].shape
        latent_vector_shape = encoder_output[0] + 6
        
        
        self.action_space = Dict({
            
                            'encoder': Box(np.tile(-1, encoder_output), np.tile(1, encoder_output)), # TODO -> I guess grid search over best size
                            'generator': Box(np.tile(-1, future[0]*future[1]), np.tile(1, future[0]*future[1])),
                            'discriminator': Box(np.array(0), np.array(1))
            })

        self.observation_space = Dict({
            
                            'encoder': Box(np.tile(-10, lags), np.tile(10, lags), dtype=np.float64),
                            'generator': Box(np.tile(-10, latent_vector_shape), np.tile(10, latent_vector_shape), dtype=np.float64),
                            
                            'discriminator': Box(np.tile(-10, (lags[0]+future[0], lags[1])), np.tile(10, (lags[0]+future[0], lags[1])), dtype=np.float64)
            })    


    def reset(self, *, seed=None, options=None):

        
        self.Done = {'__all__': False}
        
        self.index = np.random.choice(len(self.true_obs))
        self.past = self.true_obs[self.index]
        self.obs = {'encoder': self.preprocess_data(self.past)}

        return self.obs, {}
    
    def step(self, action):
        
                
        if 'encoder' in self.obs.keys():
            
            self.encoder_turn(action['encoder'])   
            
        elif 'generator' in self.obs.keys():
            
            self.generator_turn(action)
            
        elif 'discriminator' in self.obs.keys():
            
            self.discriminator_turn(action['discriminator'])
            self.Done = {'__all__': True}
        
        
        return self.obs, self.reward, self.Done, self.truncated,{}
    
    
    def encoder_turn(self, action):

        latent_vector = self.bonus_latent(action)
              
        self.obs = {'generator': latent_vector}
        max_dist = np.diff(np.sort(action)).sum() * 0.1
        
        
        self.reward = {'encoder': max_dist} 
        
    def generator_turn(self, action):
        
        reward_generator = 0
        reward_extremes = 0
        reward_mae = 0
        for i in range(self.true_obs[0].shape[1]):
            true_values = self.preprocess_data(self.true_returns[self.index][:, i]).flatten()
            
            len_lags = self.true_returns[0].shape[0]
            simulated = action['generator'][len_lags*i:len_lags*i+len_lags].flatten()
            
            #reward_generator -= wasserstein_distance(true_values, simulated)
            
            # ? I want low error for low and high + I want them to be ordered 
            #reward_extremes -= abs(true_values.min() - simulated.min()) + abs(true_values.max() - simulated.max())
            #reward_mae -= mean_squared_error(true_values, simulated)
            reward_generator -= dtw_distance(true_values, simulated)
        
        self.reward = {
                        'generator': reward_generator,
                        'encoder': reward_generator
                       }
        
        self.get_discriminator_sample(action)
        
    def discriminator_turn(self, action):
        
        discriminator_reward = cross_entropy(self.discriminator_target, action)
        generator_reward = cross_entropy(self.discriminator_target, 1 - action)
        
        if self.discriminator_target:
            self.reward = {'discriminator': discriminator_reward}
            
        else:
            self.reward = {
                            'discriminator': discriminator_reward,
                            'generator': generator_reward / 20
                        }
    
    def get_discriminator_sample(self, action):
        
        past = self.preprocess_data(self.true_obs[self.index])
        self.discriminator_target = np.random.choice([0, 1])
        
        if self.discriminator_target:
            future = self.preprocess_data(self.true_returns[self.index])
            
        else:
            future = action['generator'].reshape(self.true_returns[0].shape)
        
        concat = np.vstack([past, future])
        self.obs = {'discriminator': concat}
        
    def preprocess_data(self, data):
        
        new_d = copy.deepcopy(data)
        
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
            
                new_d[:, i] = ((data[:, i] - data[:, i].mean()) / (3 * data[:, i].std()))
            
        else:
            new_d = ((data - data.mean()) / (3 * data.std()))
        
        
        return np.clip(new_d, -3, 3)
    
    def bonus_latent(self, action):
        
        past_df = pd.DataFrame(self.preprocess_data(self.past))
        skewness = past_df.skew().clip(-5, 5)
        kurtosis = past_df.kurt().clip(-5, 5)        
        
        corr = np.corrcoef(self.past, rowvar=False)
        corr = np.append(corr[1:, 0], corr[2, 1])
        
        
        
        
        math_latent = np.stack([corr, skewness]).flatten()
        all_latent = np.append(math_latent, action)
        
        return all_latent
        
        # TODO -> Add others and then concat to encoder output

#%%          
if __name__ == '__main__':
    
    import yfinance as yf 
    from numpy.lib.stride_tricks import sliding_window_view
    
    tickers_df = yf.Tickers(['AAPL', 'GOOG', 'MSFT']).history(start='2010-01-01') 
    tickers_df = tickers_df['Close']#.pct_change().dropna()
    tickers_df
    #%%
    len_lags = 40
    len_future = 10
    true_obs = sliding_window_view(tickers_df, len_lags, axis=0).swapaxes(1, 2)[:-len_future]
    true_returns = sliding_window_view(tickers_df, len_future, axis=0).swapaxes(1, 2)[len_lags:]
    
    print(true_obs[10])
    print(true_returns[0])
    
    #%%

    print(true_obs[0].shape)   
    #%%
    market = AdverserialAutoEncoder({
        'true_obs': true_obs,
        'true_returns': true_returns
    
    })
    market.reset()
    
    #print(market.observation_space_sample(['encoder', 'decoder', 'generator', 'discriminator']))
    #print(market.past)
    
    plt.plot(market.preprocess_data(market.past))
#%%

    action = np.array([np.random.normal(0, 0.01), 0.02, 2])
    for i in range(50):
        

        obs, reward, done, dict = market.step(action)
        print(obs)
        
        if 'meta' in obs.keys():
            action = np.array([np.random.normal(0, 0.01), 0.02, 2])
            
        else:
            action = 1
            
        if done:
            market.reset()

# %%
    wasserstein_distance(true_obs[0].flatten(), market.action_space_sample(['generator'])['generator'])