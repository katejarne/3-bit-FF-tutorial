#################################################################
#                                                               #
#   A "3-bit Flip Flop" data set generator of samples           #
#   with adjutable parameters.                                  #
#   Task Parametrization like: David Sussillo & Omri Barak 2013 #
#   Mit License C. Jarne V. 1.0 2020                            #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.random import seed

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.random import seed

start_time = time.time()

def generate_trials(sample_size, mem_gap, seed_):

    #Parameters of the data set
    seed(None)
    move             = np.random.randint(10,50)
    first_in         = move #time to start the first stimulus   #30 #60
    stim_dur         = 20 #stimulus duration #20 #30
    stim_noise       = 0.1 #noise
    var_delay_length =50
    var_delay_length_ = 50 #change for a variable length stimulus
    var_delay_length_2 = 50
    out_gap          = 250-move #how much lenth add to the sequence duration    #140 #100
    #sample_size      = size # sample size
    rec_noise        = 0
       
    and_seed_A = np.array([[1],[-1],[0]])
    and_y            = np.array([1,-1,0])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+out_gap # -move#Sequence duration

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
        var_delay_ = np.zeros(sample_size, dtype=np.int)
        var_delay_2 = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
        var_delay_ =np.random.randint(var_delay_length_, size=sample_size) + 1
        var_delay_2 =np.random.randint(var_delay_length_2, size=sample_size) + 1


    x_train     = np.zeros((sample_size, seq_dur, 1))
    y_train     = np.zeros((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):
        first_in         = np.random.randint(10,100)
        out_t       = mem_gap+ first_in+stim_dur
        out_t2      = mem_gap+ first_in+stim_dur+50+var_delay[ii]+var_delay_[ii]
        out_t3      = mem_gap+ first_in+stim_dur+50+50+var_delay_2[ii]
    
        trial_types   = np.random.randint(3, size=sample_size)
        trial_types_2 = np.random.randint(2, size=sample_size)
        trial_types_3 = np.random.randint(2,size=sample_size)
       

        x_train[ii, first_in:first_in + stim_dur, 0] = and_seed_A[trial_types[ii], 0]
        y_train[ii, out_t:-1, 0]     = and_y[trial_types[ii]]
     
        x_train[ii, first_in+50+stim_dur+var_delay_[ii]:first_in+50 + 2*stim_dur+var_delay_[ii], 0] = and_seed_A[trial_types_2[ii], 0]
        y_train[ii, first_in+50 + 2*stim_dur+var_delay_[ii]+mem_gap:-1, 0]       = and_y[trial_types_2[ii]]

        x_train[ii, first_in+150+stim_dur+var_delay_[ii]+var_delay_2[ii]:first_in+150+stim_dur +stim_dur+var_delay_[ii]+var_delay_2[ii], 0] = and_seed_A[trial_types_3[ii], 0]

       
        y_train[ii, first_in+150+stim_dur +stim_dur+var_delay_[ii]+var_delay_2[ii]+mem_gap:-1, 0]= and_y[trial_types_3[ii]]
   
        
    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]

    y_train = y_train 
    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    print("--- %s seconds to generate Dataset---" % (time.time() - start_time))
    return (x_train, y_train, mask,seq_dur)   

#To see how is the training data set uncoment these lines

sample_size=10

x_train,y_train, mask,seq_dur         = generate_trials(sample_size,20,None) 
x_train_1,y_train_1, mask_1,seq_dur_1 = generate_trials(sample_size,20,None)
x_train_2,y_train_2, mask_2,seq_dur_2 = generate_trials(sample_size,20,None)


#print ("x",x_train)
#print ("y",y_train)

fig     = plt.figure(figsize=(12,10))
fig.suptitle("3 bit \"Flip-Flop\" Data Set Training Samples\n (amplitude in arb. units time in mS)",fontsize = 20)

for ii in np.arange(3):
    plt.subplot(3, 3, ii + 1)
    plt.title("Input and Output 1 sample: "+str(ii))
    #plt.plot(x_train[ii, :, 1],color='red',label="input Set")
    plt.plot(x_train[ii, :, 0],color='pink',label="Input A")
    plt.plot(y_train[ii, :, 0],color='grey',label=r'Target output $Q_A$')
    plt.ylim([-2.5, 2.5])
    #plt.xlim([-1, 450])
    plt.legend(fontsize= 9,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

for ii in np.arange(3):
    plt.subplot(3, 3, ii+3 + 1)
    plt.title("Input and Output 2 sample: "+str(ii))
    #plt.plot(x_train_1[ii, :, 1],color='g',label="input Set")
    plt.plot(x_train_1[ii, :, 0],color='deepskyblue',label="Input B")
    plt.plot(y_train_1[ii, :, 0],color='grey',label=r'Target output $Q_B$')
    plt.ylim([-2.5, 2.5])
    #plt.xlim([-1, 450])
    plt.legend(fontsize= 9,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

for ii in np.arange(3):
    plt.subplot(3, 3, ii+6 + 1)
    plt.title("Input and Output 3 sample: "+str(ii))
    #plt.plot(x_train_2[ii, :, 1],color='m',label="input Set")
    plt.plot(x_train_2[ii, :, 0],color='g',label="Input C")
    plt.plot(y_train_2[ii, :, 0],color='grey',label=r'Target output $Q_C$')
    plt.ylim([-2.5, 2.5])
    #plt.xlim([-1, 450])
    plt.legend(fontsize= 9,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)    


figname = "plots_ff/data_set_flip_flop_sample.png" 
plt.savefig(figname,dpi=200, bbox_inches='tight')
plt.show()

