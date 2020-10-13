###############################################
#                                             #
#   A "3-bit Flip Flop" data set generator    #
#   of 8 different states                     #
#   Task Parametrization like: David Sussillo #
#   & Omri Barak 2013                         # 
#   Mit License C. Jarne V. 1.0 2020          #
#                                             #
###############################################


import time
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
start_time = time.time()

def generate_trials(sample_size, mem_gap, seed_):

    #Parameters of the data set
    seed(None)
    move               = np.random.randint(10,50)
    first_in           = move #time to start the first stimulus   #30 #60
    stim_dur           = 20   #stimulus duration #20 #30
    stim_noise         = 0.1  #noise
    var_delay_length   =0     #50
    var_delay_length_  =0     # 50 #change for a variable length stimulus
    var_delay_length_2 =0     # 50
    out_gap            =1080  #how much lenth add to the sequence duration
    #sample_size        = size # sample size
    rec_noise          = 0
       
    and_seed_A = np.array([[1],[-1],[0]])
    and_y            = np.array([1,-1,0])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+out_gap

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

    x_train_1     = np.zeros((sample_size, seq_dur, 1))
    y_train_1     = np.zeros((sample_size, seq_dur, 1))

    x_train_2     = np.zeros((sample_size, seq_dur, 1))
    y_train_2     = np.zeros((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):
        first_in         = 10 # or random np.random.randint(10,100)
        out_t       = mem_gap+ first_in+stim_dur
        out_t2      = mem_gap+ first_in+stim_dur+50+var_delay[ii]+var_delay_[ii]
        out_t3      = mem_gap+ first_in+stim_dur+50+50+var_delay_2[ii]
    
        trial_types   = np.random.randint(3, size=sample_size)
        trial_types_2 = np.random.randint(2, size=sample_size)
        trial_types_3 = np.random.randint(2,size=sample_size)
       
      
        for k in np.arange(16):
            x_train_1[ii, first_in+k*50+stim_dur+var_delay_[ii]+k*20:first_in+k*50 + stim_dur +stim_dur+var_delay_[ii]+k*20, 0] = and_seed_A[0, 0]
            y_train_1[ii, first_in+k*50 + 2*stim_dur+var_delay_[ii]+k*20+mem_gap:-1, 0]       = and_y[0]
             
            if k==2 or k==3 or k==6 or k==7 or k==10 or k==11 or k==14 or k==15:
                x_train_1[ii, first_in+k*50+stim_dur+var_delay_[ii]+k*20:first_in+k*50 + stim_dur +stim_dur+var_delay_[ii]+k*20, 0] = and_seed_A[1, 0]
                y_train_1[ii, first_in+k*50 + 2*stim_dur+var_delay_[ii]+k*20+mem_gap:-1, 0]       = and_y[1]       

        for k in np.arange(16):
            if k%2==0:
                x_train[ii, first_in+k*50+stim_dur+var_delay_[ii]+k*20:first_in+k*50 + stim_dur +stim_dur+var_delay_[ii]+k*20, 0] = and_seed_A[0, 0]
                y_train[ii, first_in+k*50 + 2*stim_dur+var_delay_[ii]+k*20+mem_gap:-1, 0]       = and_y[0]
            else: 
                x_train[ii, first_in+k*50+stim_dur+var_delay_[ii]+k*20:first_in+k*50 + stim_dur +stim_dur+var_delay_[ii]+k*20, 0] = and_seed_A[1, 0]
                y_train[ii, first_in+k*50 + 2*stim_dur+var_delay_[ii]+k*20+mem_gap:-1, 0]       = and_y[1]
        
        
        for k in np.arange(16):
            if k<4 or (k>7 and k<12):
                x_train_2[ii, first_in+k*50+stim_dur+var_delay_[ii]+k*20:first_in+k*50 + stim_dur +stim_dur+var_delay_[ii]+k*20, 0] = and_seed_A[0, 0]
                y_train_2[ii, first_in+k*50 + 2*stim_dur+var_delay_[ii]+k*20+mem_gap:-1, 0]       = and_y[0]
            else: 
                x_train_2[ii, first_in+k*50+stim_dur+var_delay_[ii]+k*20:first_in+k*50 + stim_dur +stim_dur+var_delay_[ii]+k*20, 0] = and_seed_A[1, 0]
                y_train_2[ii, first_in+k*50 + 2*stim_dur+var_delay_[ii]+k*20+mem_gap:-1, 0]       = and_y[1]


    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]

    y_train = y_train 
    x_train = x_train # if  noise: + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    print("--- %s seconds to generate Dataset---" % (time.time() - start_time))

    x_train_dstack= np.dstack((x_train_1,x_train,x_train_2)) #Stack arrays in sequence depth wise (along third dimension) 
    y_train_dstack= np.dstack((y_train,y_train_1,y_train_2))

    mask_dstack=np.dstack((mask,mask,mask))


    return (x_train_dstack, y_train_dstack, mask_dstack,seq_dur)   

#To see the training data set uncoment these lines.

sample_size=10

x_train,y_train, mask,seq_dur         = generate_trials(sample_size,20,None) 


fig     = plt.figure(figsize=(12,10))
fig.suptitle("3 bit \"Flip-Flop\" Data Set Training Sample\n (amplitude in arb. units time in mS)",fontsize = 20)

for ii in np.arange(3):
    plt.subplot(3, 3, ii + 1)
    plt.title("Input 1, Output 1 sample: "+str(ii))
    plt.plot(x_train[ii, :, 0],color='pink',label="input A")
    plt.plot(y_train[ii, :, 0],color='grey',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

for ii in np.arange(3):
    plt.subplot(3, 3, ii+3 + 1)
    plt.title("Input 2, Output 2 sample: "+str(ii))
    plt.plot(x_train[ii, :, 1],color='deepskyblue',label="input B")
    plt.plot(y_train[ii, :, 1],color='grey',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

for ii in np.arange(3):
    plt.subplot(3, 3, ii+6 + 1)
    plt.title("Input 3, Output 3 sample: "+str(ii))
    plt.plot(x_train[ii, :, 2],color='g',label="input C")
    plt.plot(y_train[ii, :, 2],color='grey',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)    


figname = "plots_ff/data_set_flip_flop_samples.png" 
plt.savefig(figname,dpi=200)
plt.show()

