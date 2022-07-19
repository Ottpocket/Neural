import tensorflow as tf
import tensorflow_addons as tfa

def Batch_Drop_Dense(x, name, drop_rate, layer_size, activation = 'relu'):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(layer_size, activation=activation), name= f'Dense_{name}')(x)
    #x = tf.keras.layers.Dense(layer_size, activation=activation, name= f'Dense_{name}')(x)
    return x

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


class EmbeddingLayer(tf.keras.layers.Layer):
    '''
    Implementation of the Embedding layer.  Embeds all features into `num_dims`
    dimensions.  Takes in (None, FEATURES) tensor and outputs (None, FEATURES * num_dims) size tensor.

    ARGUMENTS
    _____
    num_dims: (int) the number of embedded dimensions.  If None, skips embedding

    Application
    ______________
    EL = EmbeddingLayer(3)
    x = tf.reshape(tf.range(0,10, dtype=tf.float32), (5,2))
    print(x.numpy())

    y = EL(x)
    print(y.numpy())
    '''
    def __init__(self, num_dims=None, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.num_dims = num_dims
        if self.num_dims is not None:
            self.emb = tf.keras.layers.Conv1D(filters=self.num_dims, kernel_size=1, activation='relu')
            self.Flatten = tf.keras.layers.Flatten()

    def get_config(self):
        config = super(EmbeddingLayer, self).get_config()
        config.update({"num_dims": self.num_dims})
        return config

    def call(self, inputs):
        if self.num_dims is None:
            return inputs

        return self.Flatten(self.emb(tf.expand_dims(inputs, -1)))


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


class NoiseMaker(tf.keras.layers.Layer):
    '''
    Randomly selects between several different noise types.
    
    ARGUMENTS
    -------------------
    gauss: (float or None) variance parameter of gaussian layer.  Noise is additive.  `None` deselects the layer.
    mixup: (float or None) alpha of mixup.  Prob you keep the original input.  `None` deselects the layer.
    cutmix: (float or None) lambda of cutmix.  Prob you keep the original input.  `None` deselects the layer.
    dropout: (float or None) prob of dropout.  Prob you drop the original input for 0.  `None` deselects the layer.
    
    OUTPUTS
    -------------------
    NoiseMaker layer
    '''
    def __init__(self, gauss=.01, mixup =.9, cutmix=.9, dropout=.1, **kwargs):
        super(NoiseMaker, self).__init__(**kwargs)
        
        self.noise_dict = {}
        if (gauss is not None) and (gauss <1) and (gauss>0):
            self.noise_dict['gauss'] = tf.keras.layers.GaussianNoise(gauss)
        
        if (mixup is not None) and (mixup <1) and (mixup>0):
            self.noise_dict['mixup']  = MixUp(mixup)
        
        if (cutmix is not None) and (cutmix <1) and (cutmix>0):
            self.noise_dict['cutmix']  = CutMix(cutmix)
    
        if (dropout is not None) and (dropout <1) and (dropout>0):
            self.noise_dict['dropout']  = tf.keras.layers.Dropout(dropout)
        
        
    def call(self, inputs):
        specific_noise = np.random.choice(list(self.noise_dict.keys()), size=1)[0]
        print(specific_noise)
        x = self.noise_dict[specific_noise](inputs)
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
