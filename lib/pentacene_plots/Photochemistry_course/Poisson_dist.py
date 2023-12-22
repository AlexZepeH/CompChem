import matplotlib.pyplot as plt
import math
import numpy as np

def math_function(n,z):
    return (z^n)/(math.factorial(n))*math.exp(-z)

z_values=[1,2,3,4]
x_data=np.arange(0,10,1)

for z in z_values:
    plt.plot(x_data,[math_function(n, z) for n in x_data],label=f'z={z}',marker='x')


plt.xlabel('n')
plt.ylabel('f(n,z)')
plt.legend()
plt.show()