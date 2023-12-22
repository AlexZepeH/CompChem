#!/usr/bin/python3
from functools import partial
import inspect
import  numpy, scipy.linalg, math, os, scipy, string
from scipy.optimize import minimize

# input templates
class InputTemplate(string.Template): delimiter = '@'
#template_gaussian = InputTemplate(open('/home/alejandro/templates/template_scan_ot4dim.dat').read())
template_gaussian = InputTemplate(open('/home/alejandro/templates/planar_ot4dim.gjf').read())
template_sbatch   = InputTemplate(open('/home/alejandro/templates/template_submit_t-scan.sh').read())

class Scan_Template():
    def __init__(self, scan_points, file_names, label_names, *others):
       self.scan_points=scan_points
       self.file_names=file_names
       self.label_names=label_names

    def make_input_with(self, *others, coord_value, name, label, 
                             xyz=False, psi=False, gau=False, no_file=False, append=False):
       if gau:
          log = template_gaussian.substitute(DIVALUE=coord_value, LABEL=label, 
                                             CHK=os.path.basename(name+'.chk'))                 
          inp = open(name+'.gjf','w')
          inp.write(log)
          inp.close()
          #
          log = template_sbatch.substitute(INP=os.path.basename(name+'.gjf'), 
                                           JOB=os.path.basename(name),
                                           DIR= "@tc148:/$"+"{WORK_SCR}")
          inp = open(name+'.sh','w')
          inp.write(log)
          inp.close()
     
class Elec_Structure():
    def __init__(self,n,file_list,f_type):
        self.f_type=f_type
        self.file_list=file_list
        self.n=n

def get_indent(astr):

    """Return index of first non-space character of a sequence else False."""

    try:
        iter(astr)
    except:
        raise

    # OR for not raising exceptions at all
    # if hasattr(astr,'__getitem__): return False

    idx = 0
    while idx < len(astr) and astr[idx] == ' ':
        idx += 1
    if astr[0] != ' ':
        return False
    return idx
def get_float_number(astr):

    """Return index of first number/point character of a sequence else False."""

    try:
        iter(astr)
    except:
        raise

    # OR for not raising exceptions at all
    # if hasattr(astr,'__getitem__): return False

    idx = 0
    while idx < len(astr) and astr[idx] in '0123456789.-':
        idx += 1
    if astr[0] not in '0123456789.-':
        return False
    return idx
      
def extract_text(input_file, output_file, search_string, n,gap, next_non_space_character = False,instance=1):
    if not next_non_space_character:
        try:
            with open(input_file, 'r') as f:
                text = f.read()

            index = text.find(search_string)
            if index != -1:
                extracted_text = text[index + len(search_string) + gap :index + gap+1 + len(search_string) + n]

                print(f"Extracted text: {extracted_text}")
                return extracted_text
            else:
                print("Search string not found in the input file.")

        except FileNotFoundError:
            print("Input file not found.")
    else:
        try:
            with open(input_file, 'r') as f:
                text = f.read()
            ocurrence = 1
            index = text.find(search_string)
            while ocurrence < instance:
                index = text.find(search_string,index+len(search_string))
                ocurrence += 1
            print(index)
            if index != -1:
                gap = get_indent(text[index+len(search_string):])
                #print(gap)
                n = get_float_number(text[index+len(search_string)+get_indent(text[index+len(search_string):]):])
                #print(n)
                extracted_text = text[index + len(search_string) + gap : index + gap + 1 + len(search_string) + n]

                print(f"Extracted text: {extracted_text}")
                return extracted_text
            else:
                print("Search string not found in the input file.")

        except FileNotFoundError:
            print("Input file not found.")

def modeling(func,params):
    """
    Call a function with an expansion of the array as arguments.

    Parameters:
    - func: The function to call.
    - popt: The array of parameters.

    Returns:
    - result: The result of calling the function.
    """
    result = func(*params)
    return result

def create_partial_with_last_n_arguments(func, *args):
    """
    Inputs:
    func of the form func(variable, *params) -> returns the computation
    *args, a list of values (numerical)

    Output:
    returns a parametrized model, where the parameters are the values passed on thorugh *args and correspond in a 1-to-1 relation to the *params of func
    """

    # Get the parameter names of the function and assigns them to an numpy array
    param_names = inspect.signature(func).parameters.keys()
    np_param_names = numpy.array(list(param_names))
    # Determine the number of parameters the function expects
    num_parameters = len(param_names)

    # Check if there are enough arguments
    if len(args) < num_parameters-1:
        raise ValueError(f"Not enough arguments provided. Function '{func.__name__}' expects {num_parameters} arguments.")


    # Extract the last n-1 arguments
    last_n_minus_one_arguments = args[-(num_parameters-1):]

    if len(param_names)-1 != len(last_n_minus_one_arguments):
        raise ValueError("Arrays must be of the same length.")
    
    #creates a dictionary with key value pairs constructed from the name of the *params of func and the passed values in *args
    arguments_dict = dict(zip(np_param_names[1:],last_n_minus_one_arguments))

    # Create the partial function with the last n-1 arguments
    partial_function = partial(func, **arguments_dict)

    return partial_function

def save_coeffs(*coeffs, out='/home/alejandro/tutorial/scan3/coeff_diab.dat'):
    "Save back to the c6,c5,c4,c3,c2,c1,c0 order."
    nc = len(coeffs)
    data = numpy.zeros((coeffs[0].size, nc))
    for i in range(nc):
        data[:,i] = coeffs[i][::-1]
    outf = open(out, 'w')
    for i in range(len(data)): 
        outf.write(nc*"%14.6E" % tuple(data[i]))
        outf.write("\n")
    outf.close()

def save_adiab_coeffs(params_v1,params_v2,out='adiab_coeffs.dat'):
        # Create an array of tuples
    tuple_array = numpy.column_stack((params_v1, params_v2))

    # Write tuples to a .dat file
    with open(out, "w") as file:
        for tuple_row in reversed(tuple_array):
            file.write("\t".join(map(str, tuple_row)) + "\n")

def poly_model(x,c0,c1,c2,c3,c4,c5,c6):
    return c0+c1*x+c2*x**2+c3*x**3+c4*x**4+c5*x**5+c6*x**6

def modeling(data,func,params):
    """
    Call a function with an expansion of the array as arguments.

    Parameters:
    - func: The function to call.
    - popt: The array of parameters.

    Returns:
    - result: The result of calling the function.
    """
    x_fit = numpy.linspace(min(data), max(data), 100)
    y_fit = func(x_fit, *params)
    return x_fit,y_fit

def t_data_in_eV(raw_scan,in_rad=False,original_data=True):
    ev2Hartree = 1/27.2114
    angtorad = 2*math.pi/360 
# Transposed data in eV
# --------------------------------------------------
    raw_scan_t=numpy.transpose(raw_scan)
    ev_scan = numpy.copy(raw_scan_t)
    ev_scan[1]= ev_scan[1]/ev2Hartree
    vg0_min = numpy.min(ev_scan[1])
    ev_scan[1]-= vg0_min 
    ev_scan[2]+= ev_scan[1]
    ev_scan[3]+= ev_scan[1]
    if original_data:
        ev_scan[0][21:]-=360
    if in_rad:
        ev_scan[0] = ev_scan[0]*angtorad
    # Define the column index to use as the sorting key
    sorting_column = 1  # In this case, we're sorting based on column 1 (0-indexed)

    # Get the sorting indices for the first row (index 0)
    sorting_indices = numpy.argsort(ev_scan[0])

    # Use advanced indexing to sort the entire array based on the sorting indices
    sorted_ev_scan = ev_scan[:, sorting_indices]
    return sorted_ev_scan

