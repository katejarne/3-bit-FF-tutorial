# 3-bit-FF-tutorial

# TUTORIAL for the premiere

https://arxiv.org/abs/2010.07858

Soft ver: Python 3.6.9
TF:2.0.0
Keras: 2.3.1

#####################################

# An example of a trained network is provided (with all 20 train instances) to generate the plots of RNN from paper, but you can train your own.

#For network training use:

loop_train_nets.py  (main)

Tune its parameters to decide the number of networks, units, and time delay.

main calls:

1) generate_data_set.py
2) binary_3bit_ff_RNN_to_loop.py

and save the trained networks at the folder:

"weights_ff" and "plot at plots_ff"

######################################

#For Plot RRN use:

load_and_plot_models.py (main)

main calls:

1) plot_collective_activity.py
2) generate_data_set_cube.py to plot the 8-memory states (or generate_data_set.py for single transitions)

The matrix to analyze must be at "matrix_weight_example" folder and plots at "plots_ff"



