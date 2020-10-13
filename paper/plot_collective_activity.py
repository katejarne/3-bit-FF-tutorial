################################################
#                                              #
#  Load and plot network activity with         #
#  Dimentionally Reduction                     #
#  Mit License C. Jarne V. 1.0 2020            #
#                                              #
################################################


# Code for plot Activity when different input are applied.
# Plot of Individual neural state for the interation that you defined.
# Plot of SVD in 2 and 3D
# Plot of PCA in 3D

#OS
import time

#Numpy
import numpy as np

#Scipy
from scipy import signal

#Matplotlib
import matplotlib.pyplot as plt
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


#Keras 
from keras import backend as K
from keras.models import Sequential, Model

# scikit learn (for SVD or PCA)
from sklearn.decomposition import PCA
import sklearn.decomposition


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_sample(sample_number,input_number,neurons,x_train_dstack, y_train_dstack,model,seq_dur,i,plot_dir,f,string_name):

    frecuencias                 =[]
    seq_dur                     = len(x_train_dstack[sample_number, :, 0])
    test                        = x_train_dstack[sample_number:sample_number+1,:,:]
    colors                      = cm.rainbow(np.linspace(0, 1, neurons+1))    
    y_pred                      = model.predict(test)

    ###################################

    # Status for the sample value at the layer indicated
    capa=0

    #First Layer:
    get_0_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    layer_output       = get_0_layer_output([test])[capa]
    

    #layer_output= model.layers[0].output
    #print("layer_output",layer_output)
    
    #Second Layer:
    get_1_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    #layer_output_1     = get_1_layer_output([test])[capa]
                
  
  
    layer_output_T       = layer_output.T
    print("layer_output",layer_output_T)
    array_red_list       = []

    ####################################
    
    y_pred              = model.predict(test)

    # To generate the Populational Analysis

    for ii in np.arange(0,neurons,1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        array_red_list.append(neurona_serie)

    #SVD and PCA with scikit learn  
  
    array_red = np.asarray(array_red_list)
    sdv       = sklearn.decomposition.TruncatedSVD(n_components=2)
    sdv_3d    = sklearn.decomposition.TruncatedSVD(n_components=3)
    X_2d      = sdv.fit_transform(array_red.T)
    X_3d      = sdv_3d.fit_transform(array_red.T)

    pca       = PCA(n_components=3)
    X_pca_    = pca.fit(array_red)
    X_pca     = pca.components_


    ####################################
    
    #2-Dim plots

    fig = plt.figure()
    fig.suptitle("SDV Network Population Analysis",fontsize = 20)
    ax1 = fig.add_subplot(111)
    
    ax1.plot(X_2d[:,0],X_2d[:,1],c='g',marker="p",zorder=2,label='Dim Reduction of the network')
    ax1.scatter(X_2d[0,0],X_2d[0,1],c='r',marker='^',s=70,label='start')
    ax1.scatter(X_2d[-1,0],X_2d[-1,1],c='b',marker='^',s=70,label='stop')
    plt.legend(loc='upper left',fontsize= 'x-small');
    plt.ylabel('C1')
    plt.xlabel('C2')
    plt.ylim([-3,3])
    plt.xlim([-4, 4])
    
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_sdv_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    #plt.savefig(figname,dpi=200)
    plt.close() 

    print("------------")
    ordeno_primero_x=X_pca[0]
    ordeno_primero_y=X_pca[1]
    ordeno_primero_z=X_pca[2]

    fig = plt.figure()
    fig.suptitle("PCA Network Population Analysis",fontsize = 20)
    ax1 = fig.add_subplot(111)
    
    ax1.plot(X_pca[0],X_pca[1],c='c',marker="p",zorder=2,label='Dim Reduction of the network')    
    ax1.scatter(ordeno_primero_x[0],ordeno_primero_y[0],s=70,c='r',marker="^",label='start')
    ax1.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],s=70,c='b',marker="^",label='stop')
    plt.legend(loc='upper left',fontsize= 'x-small');
    plt.ylabel('C1')
    plt.xlabel('C2')
    plt.ylim([-0.3,0.3])
    plt.xlim([-0.15, 0.15])
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_"+str(capa)+"_individual_neurons_state_"+str(i)+".png" 
    #plt.savefig(figname,dpi=200)
    plt.close()
    #pp.show()
              
    ####################################
    #3-Dim Plots

    # How many 3d angular views you want to define

    yy        = np.arange(0,180,10)
    #yy        = np.arange(0,90,10)
    #yy        = np.arange(70,80,10)

        
    for ii,kk in enumerate(yy):
        print ("ii: ",ii," kk: ",kk)
   
        fig     = plt.figure(figsize=(10,8))
        fig.suptitle("3D plot SDV Network Population Analysis",fontsize = 20)        
        ax = fig.add_subplot(111, projection='3d')


        ax.plot(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='gray',zorder=2,label="3 d plot",marker="p")

        #The the approximate vertices
        ax.scatter(X_3d[0,0],X_3d[0,1],X_3d[0,2],c='r',marker="^",label='start',s=300)
        ax.scatter(X_3d[-1,0],X_3d[-1,1],X_3d[-1,2],c='b',marker="^",label='stop',s=300)
        ax.scatter(X_3d[40,0],X_3d[40,1],X_3d[40,2],c='deepskyblue',marker="^",label='v1: (0,0,0)',s=300)
        ax.scatter(X_3d[100,0],X_3d[100,1],X_3d[100,2],c='gold',marker="^",label='v2: (1,1,1)',s=300)
        ax.scatter(X_3d[180,0],X_3d[180,1],X_3d[180,2],c='pink',marker="^",label='v3: (1,0,1)',s=300)
        ax.scatter(X_3d[240,0],X_3d[240,1],X_3d[240,2],c='green',marker="^",label='v4: (0,1,1)',s=300)
        ax.scatter(X_3d[320,0],X_3d[320,1],X_3d[320,2],c='m',marker="^",label='v5:(0,0,1)',s=300)
        ax.scatter(X_3d[380,0],X_3d[380,1],X_3d[380,2],c='y',marker="^",label='v6: (1,1,0)',s=300)
        ax.scatter(X_3d[460,0],X_3d[460,1],X_3d[460,2],c='hotpink',marker="^",label='v7: (1,0,0)',s=300)
        ax.scatter(X_3d[520,0],X_3d[520,1],X_3d[520,2],c='deeppink',marker="^",label='v8:(0,1,0)',s=300)
     
        ax.set_xlabel('comp 1 (arb. units)',size=16)
        ax.set_ylabel('comp 2 (arb. units)',size=16)
        ax.set_zlabel('comp 3 (arb. units)',size=16)
        ax.legend()
        ax.view_init(elev=10, azim=kk)
        #ax.view_init(elev=kk, azim=10)
        figname = str(plot_dir)+"/sample_"+str(sample_number)+"_sdv_3d_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+".png" 
        plt.savefig(figname,dpi=200)
        plt.close()
    
    ####################################   
    
    #kk=70        
    for ii,kk in enumerate(yy):

        fig     = plt.figure(figsize=(18,7))
        plt.subplot(3, 2, 1) 
        plt.plot(test[0,:,0],color='pink',label='Input 1')
        plt.plot(y_train_dstack[sample_number,:, 0],color='grey',linewidth=3,label='Target Output 1')  
        plt.plot(y_pred[0,:, 0], color='r',linewidth=2,label=' Output\n 25 individual states')    
        plt.legend(fontsize= 'x-small',loc=3)
        
        plt.subplot(3, 2, 3)   
        plt.plot(test[0,:,1],color='pink',label='Input  2')
        plt.plot(y_train_dstack[sample_number,:, 1],color='grey',linewidth=3,label='Target Output2')  
        plt.plot(y_pred[0,:, 1], color='r',linewidth=2,label=' Output 2')
        plt.xlim(0,seq_dur+1)    
        plt.ylim([-1.5, 1.5])
        plt.yticks([])
        #plt.ylabel('Activity [arb. units]',fontsize = 16)
        #plt.xlabel('time [mS]',fontsize = 16)
        #plt.xticks(np.arange(0,seq_dur+1,20),fontsize = 8)
        plt.legend(fontsize= 'x-small',loc=1)


        plt.subplot(3, 2, 5)     
        plt.plot(test[0,:,2],color='pink',label='Input 3')
        plt.plot(y_train_dstack[sample_number,:, 2],color='grey',linewidth=3,label='Target Output 3')  
        plt.plot(y_pred[0,:, 2], color='r',linewidth=2,label=' Output')
        plt.xlim(0,seq_dur+1)
        plt.ylim([-1.5, 1.5])
        plt.yticks([])
        #plt.ylabel('Activity [arb. units]',fontsize = 16)
        plt.xlabel('time [mS]',fontsize = 16)
        #plt.xticks(np.arange(0,seq_dur+1,20),fontsize = 8)
        plt.legend(fontsize= 'x-small',loc=1)
    
    
        fig.suptitle("Time series and PCA 3D plot",fontsize = 20)
        ax = fig.add_subplot(122, projection='3d')
        x=X_pca[0]
        y=X_pca[1]
        z=X_pca[2]
        N=len(z)

        
        ax.plot(X_3d[:,0],X_3d[:,1],X_3d[:,2],color='gray',zorder=2,label="3 d plot",marker="p")
        ax.scatter(X_3d[0,0],X_3d[0,1],X_3d[0,2],c='r',marker="^",label='start',s=300)
        ax.scatter(X_3d[-1,0],X_3d[-1,1],X_3d[-1,2],c='b',marker="^",label='stop',s=300)
        ax.set_xlabel(' 1 (arb. units)',size=16)
        ax.set_ylabel(' 2 (arb. units)',size=16)
        ax.set_zlabel(' 3 (arb. units)',size=16)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
   
        ax.set_zticks(())
        ax.view_init(elev=10, azim=kk)
        #ax.view_init(elev=kk, azim=10)
        ax.legend(fontsize= 'small')
        fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=16)
        figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_3D_"+str(capa)+"_individual_neurons_state_"+str(i)+'_'+str(kk)+"_"+str(f)+".png"
        plt.savefig(figname,dpi=300, bbox_inches = 'tight') 
        plt.close()      
    

