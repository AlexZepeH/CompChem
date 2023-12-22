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
# [1] Options settings

class ArgPArser(argparse.ArgumentParser):
    def error(self,message):
        sys.stderr.write('error: {:s}\n\n'.format(message))
        self.print_help()
        sys.exit(2)

parser= ArgPArser(description="Automatic pbasis generator.")

parser.add_argument('-l', '--label', type=str, default ='w', help='Mode label base. Default w' )

parser.add_argument('-u', '--unit', type=str, default ='cm-1', help='Unit type. Deault cm-1' )

parser.add_argument('-i', '--input', type=str, default ='freq.txt', help='Input file. Default freq.txt' )

#parser.add_argument('-nm', '--nmodes', type=int, default =1, help='Number of normal modes. Default 1' )

parser.add_argument('-o', '--output', type=str, default ='freq_param_list.txt', help='Output file name. Default freq_param_list.txt' )


options = parser.parse_args()

if len(sys.argv) == 1:
    print(__doc__)
    sys.exit(1)

def string_generator(label,unit,input):
    param_string =""
    # Open the file in read mode
    with open(input, 'r') as file:
        # Loop through each line in the file
        index = 1
        output_string=''
        for line in file:
            # Process each line (e.g., print it)
            ptemp = "{0}{1} = ".format(label,index)+line.strip()+",{0}\n".format(unit)
            index +=1  # strip() removes the newline character at the end of each line
            output_string += ptemp
    return output_string

if __name__ == '__main__':
    text = string_generator(options.label,options.unit,options.input)
    with open(options.output, 'w') as output_f:
            output_f.write(text)