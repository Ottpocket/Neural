import numpy as np
import pandas as pd
import sys
import os
from time import time

import tensorflow_addons as tfa
import tensorflow as tf
import gc
import joblib
import pickle

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

#Getting specialty stuff from my git repo
!git clone https://github.com/Ottpocket/Neural.git
sys.path.append('/kaggle/working/Neural')
from layers import *
from models import * 


train = 
test =  
ss = 
FEATURES = 
TARGET = 

np.random.seed(42)
train_percent = .8
X_msk = np.random.choice(a=[True, False], size=int(train.shape[0]), replace=True, p=[.8,.2])

X = train.loc[X_msk,FEATURES].values
y = train.loc[X_msk, TARGET].values

val_X = train.loc[~X_msk,FEATURES].values
val_y = train.loc[~X_msk, TARGET].values

def ff(num_input_columns, BLOCKS, drop_rate, cutmix_noise, mixup_alpha, optimizer, block_sizes =None):
    
    if block_sizes is None:
        block_sizes = [num_input_columns for _ in range(BLOCKS)]
    else:
        if len(block_sizes) !=BLOCKS:
            print(f'block_sizes has {len(block_sizes)} blocks.  Needs {BLOCKS}.')
    
    #Input
    inp = tf.keras.layers.Input(num_input_columns)
    x = CutMix(noise = cutmix_noise)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = ResnetBlockTabular(output_dim = block_sizes[0], name=f'Resnet_0')(x)
    x = MixUp(alpha= mixup_alpha)(x)
    
    for i in range(1,BLOCKS):
        x = ResnetBlockTabular(output_dim = block_sizes[i], name=f'Resnet_{i}')(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    
    
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])
    return model

ES = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0, patience=20, verbose=0, mode='max')
def objective(trial):
    #from https://github.com/optuna/optuna-examples/blob/main/keras/keras_integration.py
    # Clear clutter from previous session graphs.
    tf.keras.backend.clear_session()
    batch_size = trial.suggest_categorical('batch_size', [512,1024])
    epochs = 200
    
    ###################################
    # Generate our trial model.
    ###################################
    #Model Architecture specifications
    num_input_columns= len(FEATURES)
    BLOCKS = trial.suggest_int("BLOCKS", 1, 10) 
    drop_rate= trial.suggest_float("drop_rate", 0, .2, )
    
    #Sum of cutmix and mixup <=.5
    cutmix_noise= trial.suggest_float("cutmix_noise", 0., .5)
    mixup_alpha=trial.suggest_float("mixup_alpha", 0., .5 - cutmix_noise)
    
    #Model Optimizer Specifications
    #Copy pasted from https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_eager_simple.py
    #Thanks y'all!
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    
    model = ff(num_input_columns, BLOCKS, drop_rate, cutmix_noise, mixup_alpha, optimizer)

    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    model.fit(
        X,
        y,
        batch_size=batch_size,
        callbacks=[ES, TFKerasPruningCallback(trial, "val_auc")],
        epochs=epochs,
        validation_data=(val_X, val_y),
        verbose=1,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(val_X, val_y, batch_size=10000,verbose=0)
    return score[1]
  
  
  
  study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, timeout = TIMEOUT) #Optimize for 5 hours.  Let's waste our gpu quota!
#study.optimize(objective, n_trials=5)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
joblib.dump(study, "study.pkl")
#To regain this study: study = joblib.load("study.pkl") #https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-save-and-resume-studies
