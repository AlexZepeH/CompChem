import my_lib

 # Example usage
input_files_directory = "/scratch/alejandro/tutorial/scan3/"  # Replace with your input file's path
output_file = "/home/alejandro/tutorial/scan3/scan_pes.dat"  # Replace with your output file's path
search_string = "Excited State   "  # Replace with the string you're searching for
coordinate = 'dih32' #Scan coordinate
n = 6  # Replace with the number of characters you want to extract
lvls = 2 #Number of Excited state levels to extract
scanned_points=35
full_extraction = ""
partial_extraction = ''
flag = False

for scan in range(scanned_points+1):
    input_file= input_files_directory+ "ot4dim_scan_" + "%04d" % scan + ".log"
    print(input_file)
    try:
        partial_extraction = partial_extraction + my_lib.extract_text(input_file,output_file,coordinate,0,0,True,2)+"\t"
        partial_extraction = partial_extraction + my_lib.extract_text(input_file,output_file,"E(RwB97XD) =",14,2)+"\t"
        print(partial_extraction)
        i = 1
        while i <= lvls:
            partial_extraction = partial_extraction + my_lib.extract_text(input_file, output_file, search_string + str(i), 6, 23)+ "\t"
            print(partial_extraction)
            i += 1    
        partial_extraction = partial_extraction + "\n"
        print(partial_extraction)
    except TypeError as e:
        print(f"Type Error : {e}")
        flag = True 
        partial_extraction = partial_extraction + "\n"
        # full_extraction = full_extraction + my_lib.extract_text(input_file,output_file,"E(RwB97XD) =",14,2)+"\t"
    if not flag:
        full_extraction = partial_extraction
        print(full_extraction)
    flag = False

with open(output_file, 'w') as output_f:
            output_f.write(full_extraction)