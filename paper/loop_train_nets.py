##########################################################
#                 Author C. Jarne                        #
#               call loop  (ver 1.0)                     #                       
# MIT LICENCE                                            #
##########################################################

import os
import time
import numpy as np

from binary_3bit_ff_RNN_to_loop import *
start_time = time.time()
vector=[400] #units vector add  with ',' for other more for example [50,100,...]

f          ='weights_ff'
f_plot     ='plots_ff'
distancias = []

number_net= 1
for t in vector:
    for i in np.arange(0,number_net,1):
        mem_gap = 20 #time delay
        N_rec   = t
        base= f+'/'+  os.path.basename(f+'_'+str(mem_gap)+'_N_'+str(N_rec)+'_gap_'+str(i))
        base_plot= f_plot+'/'+  os.path.basename(f_plot+'_'+str(t)+'_N_'+str(i))
        dir = str(base)
        if not os.path.exists(dir):
           os.mkdir(base)
        print(str(dir))

        dir = str(base_plot)
        if not os.path.exists(dir):
           os.mkdir(base_plot)        
        print(str(dir))
    
        pepe    =and_fun(mem_gap,N_rec,base,base_plot)
        distancias.append(pepe)
print('-------------------------')
print (distancias)
print("--- %s to train the network seconds ---" % (time.time() - start_time))
