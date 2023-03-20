##########################################################
#                 Author C. Jarne                        #
#       RNN to be train for 3-bit FF task  (ver 2.0)     #
#  To be call from with a loop Function                  #                       
#                     2020                               #
# MIT LICENCE                                            #
##########################################################

#Os
import time

#Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

#Keras and TF
import keras.backend as K
from keras.models import Sequential, Model
from keras.constraints import Constraint 
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, Callback
#from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout, SimpleRNN
from keras.utils import plot_model
from keras import metrics, optimizers, regularizers
from keras.layers import Input
import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Dataset genertor to train
from generate_data_set import *

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

plot_dir="plots_ff"

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.009, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print(" Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def and_fun(t,N_rec,base,base_plot):
    lista_distancia=[]
    #Parameters

    sample_size      = 2*15050
    epochs           = 20
    p_connect        = 0.9
    mem_gap          = t

    g=1
    def my_init_rec(shape, name=None):
        value = np.random.random(shape)
        mu=0
        sigma=np.sqrt(1/(N_rec))
        value= g*np.random.normal(mu, sigma, shape)
        return K.variable(value, name=name)

    pepe =keras.initializers.RandomNormal(mean=0.0, stddev=1*np.sqrt(float(1)/float((N_rec))), seed=None)

    seed_0=None
    seed_1=None
    seed_2=None

    x_train,y_train, mask,seq_dur         = generate_trials(sample_size,mem_gap,seed_0) 
    x_train_1,y_train_1, mask_1,seq_dur_1 = generate_trials(sample_size,mem_gap,seed_1)
    x_train_2,y_train_2, mask_2,seq_dur_2 = generate_trials(sample_size,mem_gap,seed_2)


    #Proper vector stack for input
    x_train_dstack= np.dstack((x_train,x_train_1,x_train_2)) 
    y_train_dstack=np.dstack((y_train,y_train_1,y_train_2))
    mask_dstack=np.dstack((mask,mask_1,mask_2))

    #Print for debuggin
    print("x_train ",x_train.shape)
    print("y_train",y_train.shape)
    print("x_train_dstack",x_train_dstack.shape)
    print("y_train_dstack",y_train_dstack.shape)

    #Network model construction
    seed(None)
    model = Sequential()    
    model.add(SimpleRNN(units=N_rec,return_sequences=True, input_shape=(None, 3), kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', activation='tanh',use_bias=False))
    model.add(Dense(units=3,input_dim=N_rec))
    model.save(base+'/'+base_plot[-4]+base_plot[-3]+'_00_initial.hdf5')

    
    # Model Compiling:
    ADAM           = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0001)
    model.compile(loss = 'mse', optimizer=ADAM, sample_weight_mode="temporal")

    # Saving weigths
    filepath       = base+'/ff_3b_weights-{epoch:02d}.hdf5'
    callbacks      = [EarlyStoppingByLossVal(monitor='loss', value=0.009, verbose=1), ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1),]
    history        = model.fit(x_train_dstack[50:sample_size,:,:], y_train_dstack[50:sample_size,:,:], epochs=epochs, batch_size=128, callbacks = callbacks, shuffle=True,     sample_weight=None)
    
    # Model Testing: 
    x_pred = x_train_dstack[0:50,:,:]    
    y_pred = model.predict(x_pred)

    print("x_train shape:\n",x_train.shape)
    print("x_pred  shape\n",x_pred.shape)
    print("y_train shape\n",y_train.shape)
    fig     = plt.figure(figsize=(12,10))
    fig.suptitle("3 bit \"Flip-Flop\" Data Set Training Sample\n (amplitude in arb. units time in mS)",fontsize = 20)

    for ii in np.arange(3):
        plt.subplot(3, 3, ii + 1)
        plt.title("Input 1, Output 1 sample: "+str(ii))
        
        plt.plot(x_pred[ii, :, 0],color='pink',label="Input A")
        plt.plot(y_train_dstack[ii, :, 0],color='grey',label="Expected output Q_A")
        plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output ")
        plt.ylim([-2.5, 2.5])
        #plt.xlim([-1, 450])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    for ii in np.arange(3):
        plt.subplot(3, 3, ii+3 + 1)
        plt.title("Input 2, Output 2 sample: "+str(ii))
        plt.plot(x_pred[ii, :, 1],color='deepskyblue',label="Input B")
        plt.plot(y_train_dstack[ii, :, 1],color='grey',label="Expected output Q_B")
        plt.plot(y_pred[ii, :, 1], color='r',label="Predicted Output\n Distance ")
        plt.ylim([-2.5, 2.5])
        #plt.xlim([-1, 450])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    for ii in np.arange(3):
        plt.subplot(3, 3, ii+6 + 1)
        plt.title("Input 3, Output 3 sample: "+str(ii))
        plt.plot(x_pred[ii, :, 2],color='green',label="Input C")
        plt.plot(y_train_dstack[ii, :, 2],color='grey',label="Expected output Q_C")
        plt.plot(y_pred[ii, :, 2], color='r',label="Predicted Output\n Distance ")
        plt.ylim([-2.5, 2.5])
        #plt.xlim([-1, 450])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    fig.text(0.5, 0.03, 'time [mS]',fontsize=5, ha='center')
    fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=5)    
    figname = plot_dir+"data_set_flip_flop_.png" 
    plt.savefig(figname,dpi=200)

    print(model.summary())
    plot_model(model, to_file='plots_ff/model.png')
    print ("history keys",(history.history.keys()))


    fig     = plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.title('Model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figname = base_plot+"/model_loss_"+str(N_rec)+".png" 
    plt.savefig(figname,dpi=200)
 


