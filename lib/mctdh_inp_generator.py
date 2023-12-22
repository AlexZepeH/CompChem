import my_lib
import numpy as np

file_name = ""
t_final = 2000.0
t_out = 1
tpsi = 1

file_string = f'RUN-SECTION/nname ={file_name}'

"""Function that creates the tree structure. 

    Elec = binary var
    x = binary var
    number of modes = int

    In each of this branches, number of SPFs

"""
class Basis:
    def __init__(self,label,btype,size,parameters):
        self.label = label
        self.btype= btype
        self.size=size
        self.parameters=parameters
    
    def expand_params_into_string(self):
        parameters_string = ' '.join(map(str,self.parameters))
        return parameters_string

def create_normalmodes_tree(n,spfs):
    tree = ""
    return tree
def pbasis_section(list):
    for item in list:
        if item.parameters:
            section_string=f"{item.label} {item.btype} {item.size} {item.expand_params_into_string()}/n"
        else:
            section_string=f"{item.label} {item.btype} {item.size}/n"

    return section_string
def mctdh_tree(spfs_lvl0,modes = 0,spfs_modes=3,elec=True,x=True):
    if elec:
        output_string = f"0> {spfs_lvl0}"
    if x:
        output_string = output_string + f" {spfs_lvl0}"
    if modes != 0:
        output_string = output_string + f" {spfs_lvl0}"

    output_string = output_string +"/n  1> [el]/n  1> [x]"
    if modes != 0:
        create_normalmodes_tree(modes,spfs_modes)

        output_string += create_normalmodes_tree(modes,spfs_modes)
    output_string = output_string +"/nend-mlbasis-section/npbasis-section/n"
    output_string = output_string+"/n"+pbasis_section()

    