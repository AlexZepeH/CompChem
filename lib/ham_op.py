"""
Generates the freq parameter list in the operator file for the MCTDH Heidelberg university program

Things to improve:
1) (determine type of extra parameters required i.e. hoxeq,hofreq,homass or xi-xf, etc. )
2) Parameters accepted by the type of basis used


ALejandro Zepeda Herrera (azepeda1297@gmail.com)
Frankfurt am Main, Dec 1 2023

"""

# import sys
# import argparse
# [1] Options settings

# class ArgPArser(argparse.ArgumentParser):
#     def error(self,message):
#         sys.stderr.write('error: {:s}\n\n'.format(message))
#         self.print_help()
#         sys.exit(2)

# parser= ArgPArser(description="Automatic pbasis generator.")

# parser.add_argument('-l', '--label', type=str, default ='kappa', help='Root of files. Default kappa' )

# parser.add_argument('-n', '--nlevels', type=int, default =2, help='Number of excited states. Deault 2' )

# parser.add_argument('-m', '-modes', type=int, default =1, help='Number of modes. Default 1' )

# #parser.add_argument('-nm', '--nmodes', type=int, default =1, help='Number of normal modes. Default 1' )

# parser.add_argument('-o', '--output', type=str, default ='freq_param_list.txt', help='Output file name. Default eta_params.txt' )
# parser.add_argument('-u', '--units', type=str, default =1, help='Units. Default ev' )


# options = parser.parse_args()

# if len(sys.argv) == 1:
#     print(__doc__)
#     sys.exit(1)

def ham_modes(num_modes,label,elec=False,x=False):
    output_string=""
    j = 1
    flag = 0
    while j <= num_modes:
        if flag == 0:
            output_string += "modes \t| {0}{1} ".format(label,j)
        elif flag < 4:
            output_string += "| {0}{1} ".format(label,j)
        else:
            output_string += "| {0}{1}\n".format(label,j)
            flag = -1
        flag +=1
        j+=1
    return output_string

def eta_ham(n,num_modes):
    output_string=""
    i = 1
    while i <= n:
        j = 1
        while j <= num_modes: 
            output_string+="eta{0}_{1}*3.0^0.5\t|{2} q |1 S{0}&{0}\n".format(i,j,j+2)
            j +=1
        i +=1
    return output_string

def nm_potential(num_modes):
    output_string=""
    j = 1
    while j <= num_modes:
        output_string += "0.5*w{0}\t|{1} q^2\n".format(j,j+2)
        j+=1
    return output_string

def nm_vib_ham(num_modes):
    output_string=""
    j = 1
    while j <= num_modes:
        output_string += "w{0}\t|{1} KE\n".format(j,j+2)
        j+=1
    return output_string

with open('normal_modes_ham.txt', 'w') as output_f:
    output_f.write(ham_modes(132,'k'))

with open('eta_ham.txt', 'w') as output_f:
    output_f.write(eta_ham(2,132))

with open('nm_potential.txt','w') as output_f:
    output_f.write(nm_potential(132))

with open('nm_vib_ham.txt','w') as output_f:
    output_f.write(nm_vib_ham(132))

