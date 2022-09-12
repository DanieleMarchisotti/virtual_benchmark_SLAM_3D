import json
import numpy as np
import os

main_folder='C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\' \
            'Point Cloud Processing\\Laser_scanner_simulation_new\\datasets'
dataset_folders=[main_folder+"\\"+folder for folder in os.listdir(main_folder)]

json_file_text={
        "width": 640,
        "height": 480,
        "intrinsic_matrix":[
        618.2991943359375,
        0,
        0,
        0,
        618.1334228515625,
        0,
        320,
        240,
        1
    ]
    }

for folder in dataset_folders:
    with open(folder+"\\camera_intrinsic.json","w") as f:
        obj=json.dump(json_file_text,f,indent=4)
