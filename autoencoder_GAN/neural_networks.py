import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
import keras
from ray import tune

class EncoderModel(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(EncoderModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
             

        self.encoder = tf.keras.layers.Input(shape=original_space.shape, name="encoder")
        
        encoder = tf.keras.layers.Dense(64, activation='selu')(self.encoder)
        encoder = tf.keras.layers.Dense(128, activation='selu')(encoder)
        encoder = tf.keras.layers.Flatten()(encoder)

        encoder = tf.keras.layers.Dropout(0.2)(encoder)
        layer_out = tf.keras.layers.Dense(num_outputs)(encoder)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(encoder)
        
        self.base_model = tf.keras.Model([self.encoder], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        model_out, self._value_out = self.base_model(orig_obs)

        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1]) 
    
    
class GeneratorModel(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(GeneratorModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
             
        print(original_space)
        self.generator = tf.keras.layers.Input(shape=original_space.shape, name="generator")
      
        generator = tf.keras.layers.Dense(64,  activation='selu')(self.generator)
        generator = tf.keras.layers.Dense(64, activation='selu')(generator)

        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(generator)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(generator)
        
        self.base_model = tf.keras.Model([self.generator], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        model_out, self._value_out = self.base_model(orig_obs)

        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1]) 
    

class DiscriminatorModel(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DiscriminatorModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space

        self.discriminator = tf.keras.layers.Input(shape=original_space.shape, name="discriminator")
      
        discriminator = tf.keras.layers.Dense(64,  activation='selu')(self.discriminator)
        #discriminator = tf.keras.layers.Dense(64, activation='selu')(discriminator)
        discriminator = tf.keras.layers.Flatten()(discriminator)
        discriminator = tf.keras.layers.Dropout(0.3)(discriminator)

        layer_out = tf.keras.layers.Dense(num_outputs, activation='sigmoid')(discriminator)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(discriminator)
        
        self.base_model = tf.keras.Model([self.discriminator], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        model_out, self._value_out = self.base_model(orig_obs)

        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1]) 