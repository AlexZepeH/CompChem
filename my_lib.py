#!/usr/bin/python3
import  numpy, scipy.linalg, math, os, scipy, string

# input templates
class InputTemplate(string.Template): delimiter = '@'
#template_gaussian = InputTemplate(open('/home/alejandro/templates/template_scan_ot4dim.dat').read())
template_gaussian = InputTemplate(open('/home/alejandro/templates/template_ot4_dimer_opt.dat').read())
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
      
def extract_text(input_file, output_file, search_string, n,gap):
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

