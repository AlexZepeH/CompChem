import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

raw_data=np.loadtxt('/extscratch/tc163/jgreen/Pentacene/monomer/CAS-4-4/SA3-opt/cartgrad2nmgrad/nmgrad2.out')

my_data=[[273.96,616.43,843.08,1234.49,1073.81,1572.08,1528.52],[2.12,-0.31,-0.46,0.34,0.28,0.48,-1.065]]

paper_data=[[265.49,616.86,767.44,1186.08,1209.83,1409.67,1433.85,1555.90,1576.49],[0.55,0.21,0.34,0.30,0.49,0.61,0.37,0.46,0.23]]

pairs=[(0,0),(1,1),(2,2),(4,3),(3,4),(5,6),(6,8)]

own=plt.stem(my_data[0], np.abs(my_data[1]), label='Own Calculations',linefmt='r-',markerfmt=',' ,basefmt=None)
print(own)
theirs=plt.stem(paper_data[0], paper_data[1], label='Ref. paper',linefmt='b-',markerfmt=',',basefmt=None)
print(theirs)

scatter_plots = []
for item in pairs:
        listx = [np.abs(my_data[0][item[0]]),paper_data[0][item[1]]]
        listy = [np.abs(my_data[1][item[0]]),paper_data[1][item[1]]]
        paired_list = [listx,listy]
        scatter_plots.append(paired_list)

plt.legend()
#marker_cycle = cycler(marker=['o','s','x','^','D','v','*'])


marker=['o','s','x','^','D','v','*']

i=0
for m in marker:
        plt.scatter(scatter_plots[i][0],scatter_plots[i][1],marker=m)
        i+=1

# # Assign markers to stems
# for i, (own_idx, their_idx) in enumerate(pairs):
#     own[own_idx].set_marker(marker[i])
#     theirs[their_idx].set_marker(marker[i])


plt.xlabel('Frequency [cm\u207B\u00B9]',fontsize=20)
plt.ylabel('Displacement [#]',fontsize=20)
plt.xticks(fontsize=14)
plt.xticks(fontsize=18)
plt.show()

