import json
import sys
import os


def load_config():
    usefull_paths = ["configs", "predictions", "models/vectors"]
    for d in usefull_paths:
        os.makedirs(d, exist_ok = True)
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], "r", encoding="utf-8") as j:
        params = json.load(j)
        _, file_name = os.path.split(sys.argv[1])
        params['experiment_suffix'] = str(file_name)[:-5] #remove .json
    with open(sys.argv[1], "w", encoding="utf-8") as j:
        json.dump(obj=params, fp=j)
    return params