import my_lib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    raw_scan=np.loadtxt('/home/alejandro/tutorial/scan5/scan_pes.dat')
    sorted_ev_scan = my_lib.t_data_in_eV(raw_scan,False,False)

    isomer1_data = sorted_ev_scan[:,:26]
    isomer2_data = sorted_ev_scan[:,26:]

    print(isomer1_data)
    print(isomer2_data)

    i1_popt, i1_pcov = curve_fit(my_lib.poly_model, isomer1_data[0],isomer1_data[1])
    i1_popt_v1, i1_pcov_v1 = curve_fit(my_lib.poly_model, isomer1_data[0],isomer1_data[2])
    i1_poptv2, i1_pcovv2 = curve_fit(my_lib.poly_model, isomer1_data[0],isomer1_data[3])
    
    i2_popt, i2_pcov = curve_fit(my_lib.poly_model, isomer2_data[0],isomer2_data[1])
    i2_popt_v1, i2_pcov_v1 = curve_fit(my_lib.poly_model, isomer2_data[0],isomer2_data[2])
    i2_poptv2, i2_pcovv2 = curve_fit(my_lib.poly_model, isomer2_data[0],isomer2_data[3])

    i2_data_g0 = my_lib.modeling(isomer2_data[0],my_lib.poly_model,i2_popt)
    i2_data_v1 = my_lib.modeling(isomer2_data[0],my_lib.poly_model,i2_popt_v1)
    i2_data_v2 = my_lib.modeling(isomer2_data[0],my_lib.poly_model,i2_poptv2)

    i1_data_g0 = my_lib.modeling(isomer1_data[0],my_lib.poly_model,i1_popt)
    i1_data_v1 = my_lib.modeling(isomer1_data[0],my_lib.poly_model,i1_popt_v1)
    i1_data_v2 = my_lib.modeling(isomer1_data[0],my_lib.poly_model,i1_poptv2)

    i2_fixed_model_g0 = my_lib.create_partial_with_last_n_arguments(my_lib.poly_model,*i2_popt)
    print((my_lib.minimize(i2_fixed_model_g0,0).x[0],my_lib.minimize(i2_fixed_model_g0,0).fun))
    i2_fixed_model_v1 = my_lib.create_partial_with_last_n_arguments(my_lib.poly_model,*i2_popt_v1)
    print((my_lib.minimize(i2_fixed_model_v1,0).x[0],my_lib.minimize(i2_fixed_model_v1,0).fun))
    i2_fixed_model_v2 = my_lib.create_partial_with_last_n_arguments(my_lib.poly_model,*i2_poptv2)
    print((my_lib.minimize(i2_fixed_model_v2,0).x[0],my_lib.minimize(i2_fixed_model_v2,0).fun))

    my_lib.save_adiab_coeffs(i2_popt_v1,i2_poptv2,'/home/alejandro/tutorial/scan5/adiab_coeff_i2.dat')

    i1_fixed_model_g0 = my_lib.create_partial_with_last_n_arguments(my_lib.poly_model,*i1_popt)
    print((my_lib.minimize(i1_fixed_model_g0,0).x[0],my_lib.minimize(i1_fixed_model_g0,0).fun))
    i1_fixed_model_v1 = my_lib.create_partial_with_last_n_arguments(my_lib.poly_model,*i1_popt_v1)
    print((my_lib.minimize(i1_fixed_model_v1,0).x[0],my_lib.minimize(i1_fixed_model_v1,0).fun))
    i1_fixed_model_v2 = my_lib.create_partial_with_last_n_arguments(my_lib.poly_model,*i1_poptv2)
    print((my_lib.minimize(i1_fixed_model_v2,0).x[0],my_lib.minimize(i1_fixed_model_v2,0).fun))

    my_lib.save_adiab_coeffs(i1_popt_v1,i1_poptv2,'/home/alejandro/tutorial/scan5/adiab_coeff_i1.dat')
   
    plt.subplot(121)
    plt.scatter(isomer2_data[0], isomer2_data[1], label='g0 data', color='red')
    plt.scatter(isomer2_data[0], isomer2_data[2], label='v1 data',color='blue')
    plt.scatter(isomer2_data[0], isomer2_data[3], label='v2 data',color='green')
    plt.plot(i2_data_g0[0],i2_data_g0[1], color='red', label='g0')
    plt.plot(i2_data_v1[0], i2_data_v1[1], color='blue', label='v1')
    plt.plot(i2_data_v2[0], i2_data_v2[1], color='green', label='v2')
    plt.xlabel('Dihedral torsion (Degrees)')
    plt.ylabel('Energy (eV)')
    plt.legend()


    plt.subplot(122)
    plt.scatter(isomer1_data[0], isomer1_data[1], label='g0 data', color='red')
    plt.scatter(isomer1_data[0], isomer1_data[2], label='v1 data',color='blue')
    plt.scatter(isomer1_data[0], isomer1_data[3], label='v2 data',color='green')
    plt.plot(i1_data_g0[0],i1_data_g0[1], color='red', label='g0')
    plt.plot(i1_data_v1[0], i1_data_v1[1], color='blue', label='v1')
    plt.plot(i1_data_v2[0], i1_data_v2[1], color='green', label='v2')
    plt.xlabel('Dihedral torsion (Degrees)')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.show()

