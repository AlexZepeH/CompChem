#!/usr/bin/python3

"""
Generates the primitive basis section for the MCTDH Heidelberg university program

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

parser.add_argument('-l', '--label', type=str, default ='x', help='Mode label base. Default x' )

parser.add_argument('-t', '--type', type=str, default ='HO', help='basis type. Deault HO' )

parser.add_argument('-bs', '--bsize', type=int, default =32, help='HO Basis size. Default 32' )

parser.add_argument('-nm', '--nmodes', type=int, default =1, help='Number of normal modes. Default 1' )

parser.add_argument('-o', '--output', type=str, default ='pbasis.txt', help='Output file name. Default pbasis.txt' )

hoxeq=0.0
hofreq=1.0
homass=1.0

options = parser.parse_args()

if len(sys.argv) == 1:
    print(__doc__)
    sys.exit(1)

def pbasis_string_generator(nmodes,label,size,type,*kargs):
    pbasis_string =""
    for i in range(nmodes):
        ptemp = label+ str(i+1).zfill(2) + "\t{0}\t{1:n}\t{2:.1f}\t{3:.1f}\t{4:.1f}\n".format(type,size,kargs[0],kargs[1],kargs[2])
        pbasis_string+=ptemp
        ptemp=""

    return pbasis_string

if __name__ == '__main__':
    text = pbasis_string_generator(options.nmodes,options.label,options.bsize,options.type,0.0,1.0,1.0)
    with open(options.output, 'w') as output_f:
            output_f.write(text)
    
