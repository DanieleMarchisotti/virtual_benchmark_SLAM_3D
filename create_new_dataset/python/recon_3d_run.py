# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# importing required libraries
import json
import time, datetime
import sys
sys.path.append("../Utility")
from file import check_folder_structure
sys.path.append(".")
from initialize_config import initialize_config
import os

# for scene 1
analysis_folders = ["..\\analysis_folder_scene_1"]
# for scene 2
# analysis_folders = ["analysis_folder_scene_2"]


is_make=input("Make fragments (y/n)? ")
is_register=input("Register fragments (y/n)? ")
is_refine=input("Refine fragments (y/n)? ")
is_integrate=input("Integrate final scene (y/n)? ")
config_folders=[folder+"\\config" for folder in analysis_folders]
config_file_path_list=[]
for folder in config_folders:
    config_file_path_list.extend([folder+"\\"+file for file in os.listdir(folder)])

for config_file in config_file_path_list:
    with open(config_file) as json_file:
        config = json.load(json_file)
        initialize_config(config)
        check_folder_structure(config["path_dataset"])
    assert config is not None

    config['debug_mode'] = False

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))

    times = [0, 0, 0, 0]
    if is_make=="y":
        start_time = time.time()
        import make_fragments
        make_fragments.run(config)
        times[0] = time.time() - start_time
    if is_register=="y":
        start_time = time.time()
        import register_fragments
        register_fragments.run(config)
        times[1] = time.time() - start_time
    if is_refine=="y":
        start_time = time.time()
        import refine_registration
        refine_registration.run(config)
        times[2] = time.time() - start_time
    if is_integrate=="y":
        start_time = time.time()
        import integrate_scene
        integrate_scene.run(config)
        times[3] = time.time() - start_time

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()