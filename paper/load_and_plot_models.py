################################################
#                                              #
#  Load and plot network weights and activity  # 
#  Mit License C. Jarne V. 1.0 2020            #
#                                              #
################################################

#System Libraries
import os
import time
import fnmatch

#Numpy
import numpy as np
from numpy import linalg as LA

#Matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pylab
from pylab import grid

#Scipy
import scipy
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import norm, skew, kurtosis

#Keras and Tensorflow
from keras.utils import CustomObjectScope
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
#from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout, SimpleRNN
from keras import metrics, optimizers, regularizers, initializers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.constraints import Constraint
import tensorflow as tf

# Taking dataset from function cube or general data set

from generate_data_set_cube import *
#from generate_data_set import *

# To plot network status

from plot_collective_activity import *

#Parameters:

sample_size_3       = 10 # or the amount needed for example 5
mem_gap             = 20
sample_size         = 64 # Data set to print some results
lista_distancia_all =[]

# Generate a dataset to study the Network properties:

con_matrix_list_pos            = []
con_matrix_list_neg            = []
con_matrix_list                = []

seed_0=0
seed_1=1
seed_2=2

#If General data set use this:

'''
x_train,y_train, mask,seq_dur         = generate_trials(sample_size,mem_gap,seed_0) 
x_train_1,y_train_1, mask_1,seq_dur_1 = generate_trials(sample_size,mem_gap,seed_1)
x_train_2,y_train_2, mask_2,seq_dur_2 = generate_trials(sample_size,mem_gap,seed_2)


x_train_dstack= np.dstack((x_train,x_train_1,x_train_2)) #Stack arrays in sequence depth wise (along third dimension)

#x_train_dstack= np.dstack((x_train,x_train_1))
#x_train_stack = np.stack((x_train,x_train_1,x_train_2))
#x_train_hstack= np.hstack((x_train,x_train_1,x_train_2)) 

y_train_dstack=np.dstack((y_train,y_train_1,y_train_2))

#y_train_dstack=np.dstack((y_train,y_train_1))
mask_dstack=np.dstack((mask,mask_1,mask_2))

'''

#If cube data set use this:

x_train_dstack, y_train_dstack, mask_dstack,seq_dur   =generate_trials(sample_size,mem_gap,seed_0) 


test                           = x_train_dstack[0:4,:,:] # Here you select from the generated data set which is used for test status
test_set                       = x_train_dstack[0:20,:,:]
y_test_set                     = y_train_dstack[0:20,:,0]

#dir definition
r_dir   ="/home/kathy/Escritorio/Neuronal_networks/my_code_2020/3bit_ff/paper/matrix_weight_example/weights_ff_20_N_400_gap_9"
plot_dir="plots_ff"

g=1

#Figure seting

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]


print("Model Details for each file")

for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    
    for i,f in enumerate(files):
        if fnmatch.fnmatch(f, '*20.hdf5'):
           r_dir=root
           print("file: ",f)
           string_name=root[-5:]
           print("String name",string_name)
           #General network model construction:

           model = Sequential()
           model = load_model(r_dir+"/"+f)      

           print("-----------",i)

           for i, layer in enumerate(model.layers):
               print("i-esima capa: ",i )
               print(layer.get_config())

           pesos     = model.layers[0].get_weights()
           pesos__   = model.layers[0].get_weights()[0]
           pesos_in  = pesos[0]
           pesos_out = model.layers[1].get_weights()
           pesos     = model.layers[0].get_weights()[1] 
           #biases   = model.layers[0].get_weights()[2] #if net has bias

           N_rec                          = len(pesos_in[0])  # it has to match the value of the recorded trained network
           neurons                        = N_rec
           colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))

           print("-------------\n-------------")   
           units           = np.arange(len(pesos))
           conection       = pesos

           con_matrix_list.append(pesos)

           print("##########################" )      

           histo_lista    =[]
           array_red_list =[]

           peso_mask      = 0.001
           peso_mask_2    =-0.001


           conection_usar=conection

           #Matrix eigenvalue decomposition

           w, v = LA.eig(conection_usar)
           #print("eigenvalues\n", w)
           #print("eigenvectors\n",v)

           lista_dist     =np.c_[w,w.real]
           lista_dist_2   =np.c_[w,abs(w.real)]
           maximo         =max(lista_dist, key=lambda item: item[1])
           maximo_2       =max(lista_dist_2, key=lambda item: item[1])
           marcar         =maximo[0]
           marcar_2       =maximo_2[0]

           frecuency=0
           if marcar_2.imag==0:
               frecuency =0
           else: 
               frecuency =abs(float(marcar_2.imag)/(3.14159*float(marcar_2.real)))

           print( "frecuency",frecuency)


           ######### Figures
           plt.figure(figsize=cm2inch(8.5,7.5))
           t=w.real
           plt.scatter(w.real,w.imag,c=t, cmap='viridis',label="Eigenvalue spectrum\n ",s=2)
           # for plotting circle line:
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.5, color="dimgrey") # draw unit circle line
           plt.scatter(marcar.real,marcar.imag,color="red", label="Eigenvalue maximum real part",s=5)
           plt.plot([0,marcar.real],[0,marcar.imag],'-',color="grey")
           plt.axvline(x=1,color="salmon",linestyle='--')
           plt.xticks(fontsize=4)
           plt.yticks(fontsize=4)
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
           plt.xlim(-1.4,1.4)
           plt.ylim(-1.4,1.4)
           plt.legend(fontsize= 5,loc=1)
           plt.savefig(plot_dir+"/autoval_"+str(i)+"_"+str(f)+"_"+str(peso_mask)+"_"+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()
  
           ###########################################################################   
           conection_neg = np.ma.masked_where(conection_usar > peso_mask_2, conection)
           conection_pos = np.ma.masked_where(conection_usar < peso_mask, conection)
           #################################  Conectivity matrix: Negative or inhibitory connection ##################################

           plt.figure(figsize=(14,12)) 
           plt.title('Conection matrix iteration:'+str(i), fontsize = 18)#, weight="bold")
           grid(True)
           cmap = plt.cm.gist_ncar # color map
           pepe_mask     = conection_neg.compressed().shape
           pepe_mask_2   = conection_pos.compressed().shape
           pepe_sin_mask = conection.shape

           cmap.set_bad(color='white')

           plt.imshow(conection_neg,cmap='gist_ncar',interpolation="none",label='Conection matrix')

           cbar_max  = 0.75
           cbar_min  = -0.75
           cbar_step = 0.025

           cbar_2=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))       
           cbar_2.ax.set_ylabel('Weights [arbitrary unit]', fontsize = 16, weight="bold")

           plt.xlim([-1,N_rec +1])
           plt.ylim([-1,N_rec +1])
           plt.xticks(np.arange(0, N_rec +1, 20))
           plt.yticks(np.arange(0, N_rec +1, 20))
           plt.ylabel('Unit [i]',fontsize = 16)
           plt.xlabel('Unit [j]',fontsize = 16)
           #plt.legend(fontsize= 'medium',loc=1)
           plt.savefig(plot_dir+"/Neg_conection_matrix_"+str(i)+"_"+str(f)+'_'+str(peso_mask)+"_"+str(string_name)+"0.png",dpi=200)
           plt.close()
           zzz=np.zeros(50)
           conection_filt =conection_neg+conection_pos

           #################################### Conectivity matrix: positive or excitatory connections ###################################

           plt.figure(figsize=(14,12)) 
           plt.title('Conection matrix iteration:'+str(i), fontsize = 18)#, weight="bold")
           grid(True)

           cmap           = plt.cm.gist_ncar # color map

           conection[conection < peso_mask ] = 0
           conection_filt= conection
           conection_filt = np.ma.masked_where(conection_filt ==0, conection_filt)
           plt.imshow(conection_filt,cmap='gist_ncar',interpolation="none",label='Conection matrix')

           cbar_max  = 0.75
           cbar_min  = -0.75
           cbar_step = 0.025

           cbar=plt.colorbar(ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))
           cbar.ax.set_ylabel('Weights [arbitrary unit]', fontsize = 16, weight="bold")

           plt.xlim([-1,N_rec +1])
           plt.xticks(np.arange(0,N_rec +1, 20))
           plt.yticks(np.arange(0, N_rec +1, 20))
           plt.ylim([-1,N_rec +1])

           plt.ylabel('Unit [i]',fontsize = 16)
           plt.xlabel('Unit [j]',fontsize = 16)
           #plt.legend(fontsize= 'medium',loc=1)
           plt.savefig(plot_dir+"/filt_conection_matrix_"+str(i)+"_"+str(f)+"_"+str(string_name)+"_0.png",dpi=200)
           plt.close()

           ############################################

           # Model Testing: 
           x_pred = x_train_dstack[0:10,:,:]
           y_pred = model.predict(x_pred)

           #############################################

           fig     = plt.figure(figsize=(12,10))
           fig.suptitle("3 bit \"Flip-Flop\" Data Set Training Sample\n (amplitude in arb. units time in mS)",fontsize = 20)

           for ii in np.arange(3):
               plt.subplot(3, 3, ii + 1)
               plt.title("Input 1, Output 1 sample: "+str(ii))      
               plt.plot(x_train_dstack[ii, :, 0],color='pink',label="Input A")
               plt.plot(y_train_dstack[ii, :, 0],color='grey',label="Expected output")
               plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
               plt.ylim([-2.5, 2.5])
               #plt.xlim([-1, 450])
               plt.legend(fontsize= 8,loc=4)
               plt.xticks(fontsize=8)
               plt.yticks(fontsize=8)

           for ii in np.arange(3):
               plt.subplot(3, 3, ii+3 + 1)
               plt.title("Input 2, Output 2 sample: "+str(ii))
               plt.plot(x_train_dstack[ii, :, 1],color='deepskyblue',label="Input B")
               plt.plot(y_train_dstack[ii, :, 1],color='grey',label="Expected output")
               plt.plot(y_pred[ii, :, 1], color='r',label="Predicted Output")
               plt.ylim([-2.5, 2.5])
               #plt.xlim([-1, 450])
               plt.legend(fontsize= 8,loc=4)
               plt.xticks(fontsize=8)
               plt.yticks(fontsize=8)

           for ii in np.arange(3):
               plt.subplot(3, 3, ii+6 + 1)
               plt.title("Input 3, Output 3 sample: "+str(ii))
               plt.plot(x_train_dstack[ii, :, 2],color='g',label="Input C")
               plt.plot(y_train_dstack[ii, :, 2],color='grey',label="Expected output")
               plt.plot(y_pred[ii, :, 2], color='r',label="Predicted Output ")
               plt.ylim([-2.5, 2.5])
               #plt.xlim([-1, 450])
               plt.legend(fontsize= 8,loc=4)
               plt.xticks(fontsize=8)
               plt.yticks(fontsize=8)    

           fig.text(0.5, 0.03, 'Time [mS]',fontsize=5, ha='center')
           fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=5)
           figname = plot_dir+"/data_set_flip_flop_"+str(i)+"_"+str(f)+"_"+str(string_name)+".png" 
           plt.savefig(figname,dpi=200)
           
           # Plots 
           ################################### Here we plot iner state of the network with the desierd stimulus

           for sample_number in np.arange(sample_size_3):
             print ("sample_number",sample_number)
             print_sample = plot_sample(sample_number,2,neurons,x_train_dstack,y_train_dstack,model,seq_dur,i,plot_dir,f,string_name)

          ######################################


