import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_addons as tfa
import gc
from layers import *

def dae(num_input_columns, layer_size = 1000, BLOCKS = 3, drop_rate=.05, cutmix_rate=.2,
                mixup_rate=.1, num_embedded_dims=None):
    '''
    Creates a DAE based on architecture and noise from `SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training`

    ARGUMENTS:
    -----------
    num_input_columns: (int) number of input (and output) columns in model
    layer_size: (int) # of neurons in hidden layers
    BLOCKS: (int) # BN, dropout, Dense layer blocks
    drop_rate: (float [0,1)) dropout rate
    cutmix_rate: (float [0,1)) percent of input randomly cutmixed
    mixup_rate: (float [0,1)]) percent to blend embeddings.
    num_embedded_dims: (int) how many embedded dimensions do you want the embedder to have.
          If None, just skips the embedder
    '''

    inp = tf.keras.layers.Input(num_input_columns)
    x = CutMix(cutmix_rate, name='CutMix')(inp)
    x = EmbeddingLayer(num_dims=num_embedded_dims, name= 'EmbeddingLayer')(x)
    x = Batch_Drop_Dense(x, name='zero', drop_rate= drop_rate, layer_size=layer_size)
    x = MixUp(alpha = mixup_rate, name='MixUp')(x)
    for name, i in enumerate(range(BLOCKS-1)):
        x = Batch_Drop_Dense(x, name+1, drop_rate= drop_rate, layer_size=layer_size)
    x = Batch_Drop_Dense(x, layer_size=num_input_columns, activation = None, name='Final_layer', drop_rate= drop_rate, )
    model = tf.keras.Model(inputs=inp, outputs=x)
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),
                  loss='MeanSquaredError',
                  )
    return model

def ff(num_input_columns, BLOCKS = 3, drop_rate=.05, block_sizes = None):
    '''
    Creates a standard ff neural network
    '''
    if block_sizes is None:
        block_sizes = [num_input_columns for _ in range(BLOCKS)]
    else:
        if len(block_sizes) !=BLOCKS:
            print(f'block_sizes has {len(block_sizes)} blocks.  Needs {BLOCKS}.')
    inp = tf.keras.layers.Input(num_input_columns)
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(drop_rate)(x)

    for i in range(BLOCKS):
        x = ResnetBlock(x, layer_reshaped =block_sizes[i], name=f'block_{i+1}')
        x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),
                  loss='MeanSquaredError',
                  )
    return model



class NeuralWrapper:
    def __init__(self, name, kind, FEATURES, directory, TARGET, args,
                 strategy = None, rerun_on_all_data=False):
        '''
        args: (dict) dict of all model specific NN params.
              for kind=='dae': num_input_columns, layer_size, BLOCKS, drop_rate, cutmix_rate, mixup_rate, num_embedded_dims
              for kind=='ff':  num_input_columns, BLOCKS, drop_rate, block_sizes
              num_input_columns is infered from FEATURES for
        strategy: (tf.strategy) type of machine to use on model
        rerun_on_all_data: (bool) after fit on X and val, do you run fit again
                           on both fit and val?
        '''

        self.name = name
        self.kind = kind #either dae or ff
        self.FEATURES = FEATURES
        self.directory = os.path.join(directory, name)
        self.TARGET = TARGET
        self.rerun_on_all_data = rerun_on_all_data
        if strategy is not None:
            self.strategy = strategy
        else:
            # Detect hardware, return appropriate distribution strategy
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print('Running on TPU ', tpu.master())
            except ValueError:
                tpu = None

            if tpu:
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                self.strategy = tf.distribute.experimental.TPUStrategy(tpu)
            else:
                self.strategy = tf.distribute.get_strategy()



        #Create a fold subdirectory inside main directory.
        #If main directory is already created, just added the next
        # fold directory
        if not os.path.exists(self.directory):
            self.directory = os.path.join(directory, name, 'fold0')
            os.makedirs(self.directory)
        else:
            i=0
            while os.path.exists(os.path.join(self.directory,f'fold{i}')):
                i +=1
            self.directory = os.path.join(directory, name, 'fold0')
            os.makedirs(self.directory)
        self.patience = 15
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=0,
                                                                restore_best_weights=True)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.directory, monitor='val_loss', verbose=0, save_best_only=False,
                                                                   save_weights_only=True, save_freq='epoch')
        with self.strategy.scope():
            if self.kind=='dae':
                self.model = dae(**args)
                self.embedder = tf.keras.Model(inputs=self.model.inputs,
                                               outputs=self.model.get_layer(name=f'Dense_{args["BLOCKS"]-1}').output)
                self.output_cols = args['layer_size']
            else:
                self.model = ff(**args)
                self.output_cols = 1

        self.model.save(os.path.join(self.directory, 'model1.h5'))
    def fit(self, X, val, epochs=150):
        '''
        Trains the network and saves it every n epochs.
        Optionally, trains the network once with early stopping, then restarts
            at epoch 0 and retrains network  on all data with 10% more epochs.
        ARGUMENTS:
        X: (np array that has been gauss ranked) train data.  (also the y, as this is representational learning)
        val: (np array that has been gauss ranked) val data
        retrain_all_data: (bool) after fitting with early stopping, do you want to fit to train and val together?
        save_every_n_epochs: (int or None) how often do you want to save the model.  If None, never.
        '''

        callbacks = [self.early_stopping]
        H = self.model.fit(x=X[self.FEATURES].values, y=X[self.TARGET].values, validation_data = (val[self.FEATURES].values, val[self.TARGET].values),
                      callbacks = callbacks,
                      epochs=epochs, batch_size=1024)
        self.model.save(os.path.join(self.directory, 'best_noVal.h5'))
        if self.rerun_on_all_data:
            self.model = tf.keras.models.load_model(os.path.join(self.directory, 'model1.h5'),
                                               custom_objects={'CutMix': CutMix, 'EmbeddingLayer':EmbeddingLayer, 'MixUp':MixUp})
            num_epochs = len(H.history['loss']) - self.patience
            num_epochs = int(num_epochs * 1.1)

            if self.kind=='dae':
                data = pd.concat([X[self.FEATURES],val[self.FEATURES]])
                self.model.fit(x=data[self.FEATURES].values, y= data[self.FEATURES].values, epochs = num_epochs, batch_size=1024)
            else:
                data = pd.concat([X[self.FEATURES + [self.TARGET]],val[self.FEATURES + [self.TARGET]]])
                self.model.fit(x=data[self.FEATURES].values, y= data[self.TARGET].values, epochs = num_epochs, batch_size=1024)

            self.model.save(os.path.join(self.directory, 'best_all.h5'))


    def predict(self, X):
        if self.kind == 'dae':
            return self.embedder.predict(X, batch_size=10000)
        else:
            return self.model.predict(X, batch_size=10000)

    def get_output_shape(self, data):
        '''
        returns shape of output from this neural network
        '''
        return (data.shape[0]. self.output_cols)

class NrepeatsModel:
    '''
    Wrapper function around a sklearn compatable model NN.  Either Dae-like or feed forward.
    Run NN repeats times on data
    '''
    def __init__(self, arguments, repeats=5):
        '''
        ARGUMENTS
        ____________
        model: (sklearnable model function) the model to be repeated
        arguments: (dict) dictionary of model's arguments
        repeats: (int) times the model is to be repeated
        '''
        self.models = []
        self.repeats = repeats
        self.arguments = arguments
        for i in range(repeats):
            if 'NUM' not in arguments['name']:
                arguments['name'] = f'{arguments["name"]}_NUM0'
            else:    
                i = 0
                while f'NUM{i}' in arguments['name']:
                    i += 1
            arguments['name'] = f'{arguments["name"]}_NUM{i}'      
            self.models.append(NeuralWrapper(**arguments))

    def fit(self, X, val, epochs=150):

        for i in range(self.repeats):
            print(f'Training model {i+1} of {self.repeats}')
            self.models[i].fit(X, val, epochs=150)

    def predict(self, X):
        outputs = np.zeros(shape = self.models[0].get_output_shape(X), dtype=np.float32)
        for i in range(self.repeats):
            outputs += self.models[i].predict(X) / self.repeats

        return outputs
