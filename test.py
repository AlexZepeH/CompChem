import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import math
import my_lib
from functools import partial

# N_coarse=4
# E_off='nowinborhs_oldscan_HJ_Nc{}_mod'.format(N_coarse)

# Mass-freq, necessary for MFW coord representations
# --------------------------------------------------
# mass_OT4=((4*12+32+2)*4+2)*1836
# mass_inter=mass_OT4*mass_OT4/(mass_OT4+mass_OT4)

# mass_OT=(4*12+32+2)*1836
# mass_OT_chain=mass_OT*N_coarse+2*1836
# mass_inter_coarse=mass_OT_chain/2


def FS(x,c0,c1,c2,c3,c4,c5,c6):
    return c0*np.cos(x)+c1*np.cos(x)+c2*np.cos(x)+c3*np.cos(x)+c4*np.cos(x)+c5*np.cos(x)+c6*np.cos(x)
def ddx_FS(x,c0,c1,c2,c3,c4,c5,c6):
    dd_fs = c0*np.sin(x)+c1*np.sin(x)+c2*np.sin(x)+c3*np.sin(x)+c4*np.sin(x)+c5*np.sin(x)+c6*np.sin(x)
    return dd_fs
def poly_model(x,c0,c1,c2,c3,c4,c5,c6):
    return c0+c1*x+c2*x**2+c3*x**3+c4*x**4+c5*x**5+c6*x**6   
def line(x,a,b,c):
    return a*x+b+c*x**2

def modeling(data,func,params):
    """
    Call a function with an expansion of the array as arguments.

    Parameters:
    - func: The function to call.
    - popt: The array of parameters.

    Returns:
    - result: The result of calling the function.
    """
    x_fit = np.linspace(min(sorted_ev_scan[0][0:16]), max(sorted_ev_scan[0][0:16]), 100)
    y_fit = poly_model(x_fit, *params)
    return x_fit,y_fit

if __name__ == '__main__':

    raw_scan=np.loadtxt('/home/alejandro/tutorial/scan3/scan_pes.dat')
    sorted_ev_scan = my_lib.t_data_in_eV(raw_scan)

    popt, pcov = curve_fit(poly_model, sorted_ev_scan[0][1:14],sorted_ev_scan[1][1:14])
    popt_v1, pcov_v1 = curve_fit(poly_model, sorted_ev_scan[0][1:14],sorted_ev_scan[2][1:14])
    poptv2, pcovv2 = curve_fit(poly_model, sorted_ev_scan[0][1:14],sorted_ev_scan[3][1:14])
    # Plot the original data
    plt.subplot(121)
    plt.scatter(sorted_ev_scan[0][1:14], sorted_ev_scan[1][1:14], label='g0 data')
    plt.scatter(sorted_ev_scan[0][1:14], sorted_ev_scan[2][1:14], label='v1 data')
    plt.scatter(sorted_ev_scan[0][1:14], sorted_ev_scan[3][1:14], label='v2 data')
    data_g0 = modeling(sorted_ev_scan[0][1:14],poly_model,popt)
    data_v1 = modeling(sorted_ev_scan[0][1:14],poly_model,popt_v1)
    data_v2 = modeling(sorted_ev_scan[0][1:14],poly_model,poptv2)

    fixed_model_g0 = my_lib.create_partial_with_last_n_arguments(poly_model,*popt)
    print((minimize(fixed_model_g0,0).x[0],minimize(fixed_model_g0,0).fun))
    fixed_model_v1 = my_lib.create_partial_with_last_n_arguments(poly_model,*popt_v1)
    print((minimize(fixed_model_v1,0).x[0],minimize(fixed_model_v1,0).fun))
    fixed_model_v2 = my_lib.create_partial_with_last_n_arguments(poly_model,*poptv2)
    print((minimize(fixed_model_v2,0).x[0],minimize(fixed_model_v2,0).fun))

    my_lib.save_adiab_coeffs(popt_v1,poptv2,'/home/alejandro/tutorial/scan3/adiab_coeff.dat')

    # Plot the fitted curve
    plt.plot(data_g0[0], data_g0[1], color='red', label='g0')
    plt.plot(data_v1[0], data_v1[1], color='blue', label='v1')
    plt.plot(data_v2[0], data_v2[1], color='green', label='v2')
    plt.xlabel('Dihedral torsion (Degrees)')
    plt.ylabel('Energy (eV)')
    plt.legend()

    diab_coeffs=np.loadtxt('/home/alejandro/tutorial/scan3/coeff_diab.dat').T
    data_d1 = modeling(sorted_ev_scan[0][4:12],poly_model,np.flip(diab_coeffs[0]))
    data_d2 = modeling(sorted_ev_scan[0][4:12],poly_model,np.flip(diab_coeffs[1]))
    data_coupling = modeling(sorted_ev_scan[0][4:12],poly_model,np.flip(diab_coeffs[2]))
    
    ax2 = plt.subplot(122)
    plt.plot(data_coupling[0], data_coupling[1], color='purple', label='d coupling')
    plt.plot(data_d1[0], data_d1[1], color='yellow', label='d1')
    plt.plot(data_d2[0], data_d2[1], color='black', label='d2')
    #ax2.set_ylim(-10,10)
    # Add labels and legend
    plt.xlabel('Dihedral torsion (Degrees)')
    plt.ylabel('Energy (eV)')
    plt.legend()


    # Show the plot
    plt.show()


    # plt.plot(sorted_ev_scan[0],sorted_ev_scan[1],'bo',sorted_ev_scan[0],sorted_ev_scan[2],'go',sorted_ev_scan[0],sorted_ev_scan[3],'ro')
    # #plt.axis((-math.pi,2*math.pi,0,10))
    # plt.axis((-60,200,0,10))
    # plt.ylabel("Energy [eV]")
    # plt.xlabel("D9 scan [degrees]")
    # plt.legend(["E0","E1","E2"], loc=0)
    # plt.show()


    # raw_scan[:,1:]-=raw_scan[-1,1]
    # print(raw_scan)

    # print(curve_fit(FS, sorted_ev_scan[0], sorted_ev_scan[2]))
    # print(curve_fit(FS, sorted_ev_scan[0], sorted_ev_scan[3]))