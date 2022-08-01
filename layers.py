import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def Batch_Drop_Dense(x, name, drop_rate, layer_size, activation = 'relu'):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(layer_size, activation=activation), name= f'Dense_{name}')(x)
    #x = tf.keras.layers.Dense(layer_size, activation=activation, name= f'Dense_{name}')(x)
    return x


@tf.keras.utils.register_keras_serializable()
class CategoricalHeadTabular(tf.keras.layers.Layer):
    '''
    The head of a tabular model that accepts categorical (as opposed to numeric) inputs
    The network 
    1) inputs the values
    2) (optional) adds noise to the values
    3) embeds the layers to an N-dimensional space
    4) (optional) adds noise to the embedded values
    '''
    def __init__(self, col_tokens, input_noisemaker_params = {'gauss':None, 'mixup':None, 'cutmix':.9, 'dropout':None}, embedding_dims=1, 
                 embedding_noisemaker_params= {'gauss':.01, 'mixup':.9, 'cutmix':.9, 'dropout':.1}, **kwargs):
        '''
        ARGUMENTS
        ---------------------------
        col_tokens: (list) list of number of unique tokens each columns has
        input_noisemaker_params: (dict) the parameters to a NoiseMaker layer
        embedding_dims: (bool) number of dimensions to embed each value
        embedding_noisemaker_params: (dict) the parameters to a NoiseMaker layer
        '''
        super(CategoricalHeadTabular, self).__init__(**kwargs)
        #Saving the raw arguments
        self.col_tokens = col_tokens
        self.input_noisemaker_params = input_noisemaker_params
        self.embedding_dims = embedding_dims
        self.embedding_noisemaker_params = embedding_noisemaker_params
        
        #Model Layers
        self.input_noisemaker = NoiseMaker(**input_noisemaker_params)
        self.embedding_layer = EmbeddingLayerCat(col_tokens,embedding_dims)
        self.embedding_noisemaker = NoiseMaker(**embedding_noisemaker_params)
        
    def get_config(self):
        config = super().get_config()
        config['col_tokens'] = self.col_tokens
        config['input_noisemaker_params'] = self.input_noisemaker_params 
        config['embedding_dims'] = self.embedding_dims 
        config['embedding_noisemaker_params'] = self.embedding_noisemaker_params 
        return config
    
    def call(self, input):
        x = self.input_noisemaker(input)
        x = self.embedding_layer(x)
        x = self.embedding_noisemaker(x)
        return x


@tf.keras.utils.register_keras_serializable()
class CutMix(tf.keras.layers.Layer):
    '''
    Implementation of CutMix

    Args
    _____
    noise: (R in [0,1)) probability that a value is not sampled from distribution

    Application
    ____________
    CM = CutMix(.2)
    x = tf.reshape(tf.range(0,10, dtype=tf.float32), (5,2))
    print(x.numpy())

    y = CM(x,True)
    print(y.numpy())
    '''
    def __init__(self, noise, **kwargs):
        super(CutMix, self).__init__(**kwargs)
        self.noise = noise

    def get_config(self):
        config = super(CutMix, self).get_config()
        config.update({"noise": self.noise})
        return config

    def call(self, inputs, training=None):
        if training:
            shuffled = tf.stop_gradient(tf.random.shuffle(inputs))
            #print(shuffled.numpy())

            msk = tf.keras.backend.random_bernoulli(tf.shape(inputs), p=1 - self.noise, dtype=tf.float32)
            #print(msk)
            return msk * inputs + (tf.ones_like(msk) - msk) * shuffled
        return inputs


@tf.keras.utils.register_keras_serializable()
class EmbeddingLayer(tf.keras.layers.Layer):
    '''
    Base class to embed data    
    
    ARGUMENTS
    ------------------
    embedding_dims: (int) the size of the embeddings
    '''
    def __init__(self, num_columns, embedding_dims =1, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
    
        #Saving params
        self.embedding_dims = embedding_dims
        self.num_columns = num_columns
        
        #Model Layers
        self.slices = []
        self.embedders = [] 
        for index in range(self.num_columns):
            self.slices.append(tf.keras.layers.Lambda(lambda a,k=index: a[:,k], name=f"slice_{index}", dtype=tf.int32))
            self.embedders.append(self.get_specific_embedder(index))
            
        #Post processing 
        self.flatten = tf.keras.layers.Flatten(name='flatten')
    
    def get_specific_embedder(self, index):
        return tf.keras.layers.GaussianNoise(name = f'Noise_{index}', stddev=0.0001)
    
    def get_config(self):
        config = super().get_config()
        config['embedding_dims'] = self.embedding_dims
        config['num_columns'] = self.num_columns
        return config
    
    def call(self, input):
        embeddings = []
        for index in range(self.num_columns):
            temp = tf.expand_dims(tf.expand_dims(self.slices[index](input), -1), -1) #reshapes slice from (batch_size,) to (batch_size,1,1)
            temp = self.embedders[index](temp)
            embeddings.append(temp)
        
        return self.flatten(tf.concat(embeddings, axis=1))
        #return tf.concat(embeddings, axis=1)
    
    
@tf.keras.utils.register_keras_serializable()
class EmbeddingLayerCat(EmbeddingLayer):
    '''
    
    '''
    def __init__(self, col_tokens, embedding_dims =1, num_columns=None, **kwargs):
        #Saving params
        self.col_tokens  = col_tokens
        self.embedding_dims = embedding_dims
        super().__init__(num_columns = len(col_tokens), embedding_dims =embedding_dims, **kwargs)
    
    def get_specific_embedder(self, index):
        return tf.keras.layers.Embedding(self.col_tokens[index], self.embedding_dims, name=f'embedding_{index}')
    
    def get_config(self):
        config = super().get_config()
        config['col_tokens'] = self.col_tokens
        return config    
    
    
@tf.keras.utils.register_keras_serializable()
class EmbeddingLayerNum(EmbeddingLayer):
    '''
    Has 1d convolution per feature    
    
    ARGUMENTS
    ------------------
    num_columns:(int) number of columns to embed 
    embedding_dims: (int) the size of the embeddings
    
    APPLICATION
    ---------------------
    embedder = EmbeddingLayerNum(num_columns=2, embedding_dims=2)
    data = pd.DataFrame({'a':[-4444444, 1, 99999, 1], 'n':[1,2,-4444444, 1]}).astype('float32').values
    embedder(data)
    '''
    def __init__(self, num_columns, embedding_dims =1, **kwargs):
        super().__init__(num_columns, embedding_dims, **kwargs)
    
    def get_specific_embedder(self, index):
        return tf.keras.layers.Conv1D(filters=self.embedding_dims, kernel_size=1, activation='relu')
        
    def get_config(self):
        config = super().get_config()
        return config
    
    
@tf.keras.utils.register_keras_serializable()
class MixUp(tf.keras.layers.Layer):
    '''
    Implementation of MixUp

    Args
    _____
    alpha: (R in [0,1)) percentage of random sample to input  used

    Application
    ____________
    MU = MixUp(.1)
    x = tf.reshape(tf.range(0,10, dtype=tf.float32), (5,2))
    y = MU(x)
    print(x.numpy())
    print(y.numpy())
    '''
    def __init__(self, alpha, **kwargs):
        super(MixUp, self).__init__(**kwargs)
        self.alpha = alpha
        self.alpha_constant = tf.constant(self.alpha)
        self.one_minus_alpha = tf.constant(1.) - self.alpha

    def get_config(self):
        config = super(MixUp, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    def call(self, inputs, training=None):
        if training:
            shuffled = tf.stop_gradient(tf.random.shuffle(inputs))
            #print(shuffled.numpy())
            return self.alpha_constant * inputs + self.one_minus_alpha * shuffled
        return inputs

    
@tf.keras.utils.register_keras_serializable()
class NoiseMaker(tf.keras.layers.Layer):
    '''
    Randomly selects between several different noise types.
    
    ARGUMENTS
    -------------------
    gauss: (float or None) standard deviation parameter of gaussian layer.  Noise is additive.  `None` deselects the layer.
    mixup: (float or None) alpha of mixup.  Prob you keep the original input.  `None` deselects the layer.
    cutmix: (float or None) lambda of cutmix.  Prob you keep the original input.  `None` deselects the layer.
    dropout: (float or None) prob of dropout.  Prob you drop the original input for 0.  `None` deselects the layer.
    
    OUTPUTS
    -------------------
    NoiseMaker layer
    
    EXAMPLE USEAGE
    --------------------
    noisemaker = NoiseMaker()
    x = np.array(range(50)).reshape((5,10))
    x = tf.convert_to_tensor(x, dtype=tf.float32, dtype_hint=None, name=None)
    y = noisemaker(x, training=True)
    print(y)
    '''
    def __init__(self, gauss=.01, mixup =.9, cutmix=.9, dropout=.1, **kwargs):
        super(NoiseMaker, self).__init__(**kwargs)
        self.params = {'gauss':gauss, 'mixup':mixup, 'cutmix':cutmix, 'dropout':dropout}
        
        
        self.noise_dict = {}
        if (gauss is not None) and (gauss <1) and (gauss>0):
            self.noise_dict['gauss'] = tf.keras.layers.GaussianNoise(gauss)
        
        if (mixup is not None) and (mixup <1) and (mixup>0):
            self.noise_dict['mixup']  = MixUp(mixup)
        
        if (cutmix is not None) and (cutmix <1) and (cutmix>0):
            self.noise_dict['cutmix']  = CutMix(cutmix)
    
        if (dropout is not None) and (dropout <1) and (dropout>0):
            self.noise_dict['dropout']  = tf.keras.layers.Dropout(dropout)
            
    def get_config(self):
        config = super().get_config()
        for key, value in self.params.items():
            config[key] = value
        return config
        
    def call(self, inputs):
        specific_noise = np.random.choice(list(self.noise_dict.keys()), size=1)[0]
        x = self.noise_dict[specific_noise](inputs)
        return x
    
    
@tf.keras.utils.register_keras_serializable()
class NumericHeadTabular(tf.keras.layers.Layer):
    '''
    The head of a tabular model that accepts numerical (as opposed to categorical) inputs
    The network 
    1) inputs the values
    2) (optional) adds noise to the values
    3) embeds the layers to an N-dimensional space
    4) (optional) adds noise to the embedded values
    '''
    def __init__(self, input_noisemaker_params = {'gauss':None, 'mixup':.9, 'cutmix':.9, 'dropout':.1}, embedding_dims=1, 
                 embedding_noisemaker_params= {'gauss':.01, 'mixup':.9, 'cutmix':.9, 'dropout':.1}, **kwargs):
        '''
        ARGUMENTS
        ---------------------------
        input_noisemaker_params: (dict) the parameters to a NoiseMaker layer
        embedding_dims: (bool) number of dimensions to embed each value
        embedding_noisemaker_params: (dict) the parameters to a NoiseMaker layer
        '''
        super(NumericHeadTabular, self).__init__(**kwargs)
        #Saving the raw arguments
        self.input_noisemaker_params = input_noisemaker_params
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.embedding_dims = embedding_dims
        self.embedding_noisemaker_params = embedding_noisemaker_params
        
        #Model Layers
        self.input_noisemaker = NoiseMaker(**input_noisemaker_params)
        self.embedding_layer = EmbeddingLayerNum(embedding_dims)
        self.embedding_noisemaker = NoiseMaker(**embedding_noisemaker_params)
        
    def get_config(self):
        config = super().get_config()
        config['input_noisemaker_params'] = self.input_noisemaker_params 
        config['embedding_dims'] = self.embedding_dims 
        config['embedding_noisemaker_params'] = self.embedding_noisemaker_params 
        return config
    
    def call(self, input):
        x = self.input_noisemaker(input)
        x = self.batch_norm(x)
        x = self.embedding_layer(x)
        x = self.embedding_noisemaker(x)
        return x
    

def ResnetBlock(x, layer_reshaped, name=None):
    #arch from https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
    #x is non relued weights
    x = tf.keras.layers.Dense(layer_reshaped)(x)

    y = tf.keras.layers.BatchNormalization()(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(layer_reshaped)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dense(layer_reshaped)(y)
    if name is None:
        y = tf.keras.layers.Add()([x, y])
    else:
        y = tf.keras.layers.Add(name = name)([x, y])
    return y


class ResnetBlockTabular(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        '''
        output_dim: (int) dimension of output dense layer.
        NOTE: if output_dim == input_dim, this is a ResNetIdentityBlock
        '''
        super(ResnetBlockTabular, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        if self.output_dim == input_shape[-1]:
            self.Dense1 = None
        else:
            self.Dense1 = tf.keras.layers.Dense(output_dim)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(self.output_dim)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.dense3 = tf.keras.layers.Dense(self.output_dim)

    def call(self, input_tensor, training=False):
        if self.Dense1 is not None:
            input_tensor = self.Dense1(input_tensor)
        
        x = self.bn1(input_tensor)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dense3(x)

        return x + input_tensor
