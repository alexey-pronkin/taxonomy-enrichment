from collections import defaultdict
import sys
import os
import json

import numpy as np
# from gensim.models.fasttext import load_facebook_model
import fasttext
from string import punctuation
from ruwordnet.ruwordnet_reader import RuWordnet


class FasttextVectorizer:
    def __init__(self, model_path):
        # self.model = load_facebook_model(model_path)
        self.model = fasttext.load_model(model_path)
        self.vector_size = self.model.get_dimension()
        print('Model loaded')

    # -------------------------------------------------------------
    # vectorize ruwordnet
    # -------------------------------------------------------------

    def vectorize_ruwordnet(self, synsets, output_path):
        ids, vectors = self.__get_ruwordnet_vectors(synsets)
        self.save_as_w2v(ids, vectors, output_path)

    def __get_ruwordnet_vectors(self, synsets):
        ids = []
        vectors = np.zeros((len(synsets), self.vector_size))
        for i, (_id, texts) in enumerate(synsets.items()):
            ids.append(_id)
            # vectors[i, :] = self.__get_avg_vector(texts)
            vectors[i, :] = self.model.get_sentence_vector((" ".join(texts)).lower())
            if i in range(2):
                print("number",i, (" ".join(texts)).lower(), vectors[i, :])
        return ids, vectors

    def __get_avg_vector(self, texts):
        sum_vector = np.zeros(self.vector_size)
        for text in texts:
            words = [i.strip(punctuation) for i in text.split()]
            sum_vector += np.sum(self.__get_data_vectors(words), axis=0)/len(words)
        return sum_vector/len(texts)

    # -------------------------------------------------------------
    # vectorize data
    # -------------------------------------------------------------

    def vectorize_data(self, data, output_path):
        data_vectors = self.__get_data_vectors(data)
        self.save_as_w2v(data, data_vectors, output_path)

    def __get_data_vectors(self, data):
        vectors = np.zeros((len(data), self.vector_size))
        for i, word in enumerate(data):
            vectors[i, :] = self.model[word]
        return vectors

    # -------------------------------------------------------------
    # save
    # -------------------------------------------------------------

    @staticmethod
    def save_as_w2v(words: list, vectors: np.array, output_path: str):
        assert len(words) == len(vectors)
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
            for word, vector in zip(words, vectors):
                vector_line = " ".join(map(str, vector))
                w.write(f"{word.upper()} {vector_line}\n")


def process_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = f.read().lower().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, output_file)


if __name__ == '__main__':
    from helpers.utils import load_config
    
    config = load_config()
    ft_vec = FasttextVectorizer(config["vectorizer_path"])
    ruwordnet = RuWordnet(db_path=config["db_path"], ruwordnet_path=config["ruwordnet_path"], with_lemmas=False)
    noun_synsets = defaultdict(list)
    verb_synsets = defaultdict(list)
    for sense_id, synset_id, text in ruwordnet.get_all_senses():
        if synset_id.endswith("N"):
            noun_synsets[synset_id].append(text.lower())
        elif synset_id.endswith("V"):
            verb_synsets[synset_id].append(text.lower())
    exp_suffix = config["experiment_suffix"]
    os.makedirs(f"predictions/{exp_suffix}", exist_ok=True)
    os.makedirs(f"models/vectors/{exp_suffix}", exist_ok=True)
    synsets = {
        'nouns_synsets':noun_synsets,
        'verbs_synsets':verb_synsets,
              }
    for part in ["nouns", "verbs"]:
        vectors_dict = {
            f"ruwordnet_vectors_{part}_path" : f"../baselines/models/vectors/{exp_suffix}/ruwordnet_{part}_{exp_suffix}.txt",
            
            f"public_data_vectors_{part}_path" : f"../baselines/models/vectors/{exp_suffix}/{part}_public_{exp_suffix}.txt",
            f"private_data_vectors_{part}_path" : f"../baselines/models/vectors/{exp_suffix}/{part}_private_{exp_suffix}.txt",
            
            f"public_test_{part}_path": f"../data/public_test/{part}_public.tsv",
            f"private_test_{part}_path": f"../data/private_test/{part}_private.tsv",
            
            f"public_output_{part}_path": f"predictions/{exp_suffix}/predicted_public_{part}_{exp_suffix}.tsv",
            f"private_output_{part}_path": f"predictions/{exp_suffix}/predicted_private_{part}_{exp_suffix}.tsv",
        }
        
        ft_vec.vectorize_ruwordnet(synsets[f"{part}_synsets"], vectors_dict[f"ruwordnet_vectors_{part}_path"])
        process_data(vectors_dict[f"public_test_{part}_path"], vectors_dict[f"public_data_vectors_{part}_path"])
        process_data(vectors_dict[f"private_test_{part}_path"], vectors_dict[f"private_data_vectors_{part}_path"])
        
        config.update(vectors_dict) # Update: config with paths
        with open(sys.argv[1], "w", encoding="utf-8") as j:
            json.dump(obj=config, fp=j)