"""
Generates the freq parameter list in the operator file for the MCTDH Heidelberg university program

Things to improve:
1) (determine type of extra parameters required i.e. hoxeq,hofreq,homass or xi-xf, etc. )
2) Parameters accepted by the type of basis used


ALejandro Zepeda Herrera (azepeda1297@gmail.com)
Frankfurt am Main, Dec 1 2023

"""

import sys
import argparse
import numpy as np
import os

# [1] Options settings

class ArgPArser(argparse.ArgumentParser):
    def error(self,message):
        sys.stderr.write('error: {:s}\n\n'.format(message))
        self.print_help()
        sys.exit(2)

parser= ArgPArser(description="Automatic pbasis generator.")

parser.add_argument('-l', '--label', type=str, default ='kappa', help='Root of files. Default kappa' )

parser.add_argument('-n', '--nlevels', type=int, default =2, help='Number of excited states. Deault 2' )

parser.add_argument('-em', '--excited_monomers', type=int, default =1, help='Input file. Default 1' )

#parser.add_argument('-nm', '--nmodes', type=int, default =1, help='Number of normal modes. Default 1' )

parser.add_argument('-o', '--output', type=str, default ='freq_param_list.txt', help='Output file name. Default eta_params.txt' )
parser.add_argument('-u', '--units', type=str, default =1, help='Units. Default ev' )


options = parser.parse_args()

if len(sys.argv) == 1:
    print(__doc__)
    sys.exit(1)

def read_single_column_files(directory, file_prefix, num_files):
    columns = []

    n = 0
    while n <= num_files:
        file_path = os.path.join(directory, f"{file_prefix}{n}.txt")
        # Read each file into a NumPy array
        column = np.loadtxt(file_path, delimiter='\n')
        columns.append(column)
        n +=1
    # Stack the arrays horizontally to create a single 2D array
    stacked_array = np.column_stack(columns).T

    return stacked_array

def eta(kappa0,kappai):
     return (2*kappa0+kappai)/3.0

def param_generator(input,levels,unit):
    # Open the file in read mode
    output_string = ""
    n=0
    while n <= levels:
        i = 1
        for coupling in input[n]:
            eta_value = eta(input[0][i-1],coupling)
            ptemp = "eta{0}_{1} = {2:.10f}, {3}\n".format(n,i,eta_value,unit)
            i+=1
            output_string += ptemp
        i = 0
        n +=1
    return output_string
     
if __name__ == '__main__':
    kappa = read_single_column_files(os.getcwd(),options.label,options.nlevels)
    text = param_generator(kappa,options.nlevels,options.units)
    with open(options.output, 'w') as output_f:
            output_f.write(text)