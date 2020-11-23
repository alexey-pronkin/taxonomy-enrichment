import sys
import json
import codecs
import zipfile

from predict_models import BaselineModel, SecondOrderModel
from helpers.utils import load_config

def save_to_file(words_with_hypernyms, output_path, ruwordnet):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            for hypernym in hypernyms:
                f.write(f"{word}\t{hypernym}\t{ruwordnet.get_name_by_id(hypernym)}\n")

def main():
    models = {"baseline": BaselineModel, "second_order": SecondOrderModel}
    config = load_config()
    for part in ["nouns", "verbs"]:
        for phase in ["public", "private"]:

            with open(config[f"{phase}_test_{part}_path"], 'r', encoding='utf-8') as f:
                test_data = f.read().split("\n")[:-1]
            baseline = models[config["model"]](params=config, part=part, phase=phase)
            print("Model loaded")
            results = baseline.predict_hypernyms(list(test_data))
            save_to_file(results, config[f"{phase}_output_{part}_path"], baseline.ruwordnet)
# TODO zip, now zip all folder and we need file only
#             with zipfile.ZipFile(config[f"{phase}_output_{part}_path"] + ".zip", "w") as myzip:
#                 myzip.write(config[f"{phase}_output_{part}_path"])

if __name__ == '__main__':
    main()
