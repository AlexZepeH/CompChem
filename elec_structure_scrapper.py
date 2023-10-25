import my_lib

 # Example usage
input_files_directory = "/scratch/alejandro/tutorial/scan/"  # Replace with your input file's path
output_file = "/home/alejandro/tutorial/scan/inp_hamiltonian.dat"  # Replace with your output file's path
search_string = "Excited State   "  # Replace with the string you're searching for
coordinate = 'D9                 ' #Scan coordinate
n = 6  # Replace with the number of characters you want to extract
lvls = 2 #Number of Excited state levels to extract
scanned_points=12
full_extraction = ""

for scan in range(scanned_points+1):
    input_file= input_files_directory+ "ot4dim_scan_" + "%04d" % scan + ".log"


    full_extraction = full_extraction + my_lib.extract_text(input_file,output_file,coordinate,7,0)+"\t"
    full_extraction = full_extraction + my_lib.extract_text(input_file,output_file,"E(RwB97XD) =",14,2)+"\t"
    i = 1
    while i <= lvls:
        full_extraction = full_extraction + my_lib.extract_text(input_file, output_file, search_string+ str(i), 6, 23)+ "\t"
        i += 1    
    full_extraction = full_extraction + "\n"

with open(output_file, 'w') as output_f:
            output_f.write(full_extraction)
