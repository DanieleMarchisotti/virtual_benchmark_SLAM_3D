import json


def from_realsense(folderPathSaved):
    cx = 0
    cy = 0
    fx = 0
    fy = 0
    width = 0
    height = 0
    for file in folderPathSaved:
        try:
            with open(file) as f:
                intrinsic=json.load(f)
            cx = intrinsic["intrinsic_matrix"][6]
            cy = intrinsic["intrinsic_matrix"][7]
            fx = intrinsic["intrinsic_matrix"][0]
            fy = intrinsic["intrinsic_matrix"][4]
            width = intrinsic["width"]
            height = intrinsic["height"]
        except:
            pass
    return cx,cy,fx,fy,width,height


def from_mynteye(folderPathSaved):
    cx = 0
    cy = 0
    fx = 0
    fy = 0
    width = 0
    height = 0
    for file in folderPathSaved:
        try:
            with open(file) as f:
                intrinsic=json.load(f)
            cx = intrinsic["intrinsic_matrix"][6]
            cy = intrinsic["intrinsic_matrix"][7]
            fx = intrinsic["intrinsic_matrix"][0]
            fy = intrinsic["intrinsic_matrix"][4]
            width = intrinsic["width"]
            height = intrinsic["height"]
        except:
            pass
    return cx,cy,fx,fy,width,height
