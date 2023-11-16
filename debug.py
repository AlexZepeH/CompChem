import numpy as np
from functools import partial
import inspect
import my_lib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# # Sample numpy array
# my_array = np.array([[5, 2, 8],
#                      [1, 9, 4],
#                      [7, 6, 3]])

# # Define the column index to use as the sorting key
# sorting_column = 1  # In this case, we're sorting based on column 1 (0-indexed)

# # Get the sorting indices for the first row (index 0)
# sorting_indices = np.argsort(my_array[0])

# # Use advanced indexing to sort the entire array based on the sorting indices
# sorted_array = my_array[:, sorting_indices]

# # Print the sorted array
# print(sorted_array)

# def create_fixed_parameters_function(*fixed_parameters):
#     def my_model_function(variable):
#         # Your mathematical model logic here
#         # For illustration purposes, let's say the model is a simple sum
#         return variable + sum(fixed_parameters)
    
#     return my_model_function

# # Create a new function with fixed parameters
# fixed_parameters_model = create_fixed_parameters_function(2, 3, 4)  # Replace with your desired fixed parameters

# # Call the new function with only the variable
# result = fixed_parameters_model(5)  # Replace 5 with your desired variable
# print(result)

def create_partial_with_last_n_arguments(func, *args):
    # Get the parameter names of the function
    param_names = inspect.signature(func).parameters.keys()
    print(param_names)
    np_param_names = np.array(list(param_names))
    print(np_param_names)
    # Determine the number of parameters the function expects
    num_parameters = len(param_names)
    print(len(param_names))

    # Check if there are enough arguments
    if len(args) < num_parameters-1:
        raise ValueError(f"Not enough arguments provided. Function '{func.__name__}' expects {num_parameters} arguments.")


    # Extract the last n-1 arguments
    last_n_minus_one_arguments = args[-(num_parameters-1):]

    if len(param_names)-1 != len(last_n_minus_one_arguments):
        raise ValueError("Arrays must be of the same length.")
    
    arguments_dict = dict(zip(np_param_names[1:],last_n_minus_one_arguments))

    # Create the partial function with the last n-1 arguments
    partial_function = partial(func, **arguments_dict)

    return partial_function

# Example usage:
def FS(x, c0, c1, c2, c3, c4, c5, c6):
    # Your mathematical model logic here
    # For illustration purposes, let's say the model is a polynomial
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6

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

def poly_model(x,c0,c1,c2,c3,c4,c5,c6):
    return c0+c1*x+c2*x**2+c3*x**3+c4*x**4+c5*x**5+c6*x**6   

raw_scan=np.loadtxt('/home/alejandro/tutorial/scan3/scan_pes.dat')
sorted_ev_scan = my_lib.t_data_in_eV(raw_scan)

popt, pcov = curve_fit(poly_model, sorted_ev_scan[0],sorted_ev_scan[1])
popt_v1, pcov_v1 = curve_fit(poly_model, sorted_ev_scan[0],sorted_ev_scan[2])
poptv2, pcovv2 = curve_fit(poly_model, sorted_ev_scan[0],sorted_ev_scan[3])
# Plot the original data
plt.scatter(sorted_ev_scan[0], sorted_ev_scan[1], label='g0 data')
plt.scatter(sorted_ev_scan[0], sorted_ev_scan[2], label='v1 data')
plt.scatter(sorted_ev_scan[0], sorted_ev_scan[3], label='v2 data')
data_g0 = modeling(sorted_ev_scan[0],poly_model,popt)
data_v1 = modeling(sorted_ev_scan[0],poly_model,popt_v1)
data_v2 = modeling(sorted_ev_scan[0],poly_model,poptv2)
plt.show()
