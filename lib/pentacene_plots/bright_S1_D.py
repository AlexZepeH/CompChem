import numpy as np
import matplotlib.pyplot as plt

raw_data=np.loadtxt('/extscratch/tc163/jgreen/Pentacene/monomer/CAS-4-4/SA3-opt/cartgrad2nmgrad/nmgrad2.out')
# Get indices to sort by decreasing absolute value of the third column
indices = np.argsort(np.abs(raw_data[:, 2]))[::-1]

# Use the indices to rearrange the entire array
sorted_array = raw_data[indices]

data_to_plot = sorted_array.T
print(data_to_plot)

bars = plt.bar(data_to_plot[0][0:20].astype(str),data_to_plot[2][0:20],color='black',alpha=1.0, width=0.1, edgecolor='black', linewidth=0.4)
# Highlight specific bars with a different color
highlighted_bars = [0,16,11,15,2,10,18]
for idx in highlighted_bars:
    bars[idx].set_color('red')

plt.xlabel('frequency',fontsize=20)
plt.ylabel('Displacement',fontsize=20)
plt.title('Freq by decreasing magnitude of displacement', fontsize=18)
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)


plt.show()