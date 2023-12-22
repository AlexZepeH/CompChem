import numpy as np
import matplotlib.pyplot as plt

raw_data=np.loadtxt('/scratch/alejandro/tutorial/pentacene_132modes/adiabatic/adpop.dat')
# Get indices to sort by decreasing absolute value of the third column

# Use the indices to rearrange the entire array

data_to_plot = raw_data.T

plt.scatter(data_to_plot[0],data_to_plot[1],label='S6')
plt.scatter(data_to_plot[0],data_to_plot[2],label='S7')
# Highlight specific bars with a different color
plt.xlabel('Time [fs]',fontsize=20)
plt.ylabel('Adiabatic pop',fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


plt.show()