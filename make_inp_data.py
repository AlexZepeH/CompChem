import my_lib
import numpy as np

working_path = '/home/alejandro/tutorial/scan_2/'
di_angles = np.arange(-135.680,-200.680,-5.0)
label_list = []
file_names = []
for i in range(len(di_angles)):
    label_item = "%04d" % i
    file_names.append("ot4dim_scan_" + label_item)
    label_list.append("scan_"+ label_item)


print(di_angles)
print(label_list)
print(file_names)                                   

scan_config = my_lib.Scan_Template(di_angles,file_names,label_list)

for i in range(len(di_angles)):
    scan_config.make_input_with(coord_value=di_angles[i],name=working_path+file_names[i],label=label_list[i], gau=True)