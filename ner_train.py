import os
import pickle
import random
import subprocess
from pathlib import Path

import numpy as np
import spacy
from spacy.cli.download import download as spacy_model_download
from spacy.cli.init_config import fill_config as spacy_fill_config
from spacy.cli.train import train as spacy_train
from spacy.tokens import DocBin
from spacy.util import Config
from torch import cuda
from ...intitalizer import FUNC_NAME, trainlogger
class TrainNer:

    def __init__(self) -> None:
        self.python_path = subprocess.check_output("which python", shell=True).strip()
        self.python_path = self.python_path.decode('utf-8')
        self.PYTHON_PATH = self.python_path
        self.model_size_dict = {
            "small" : "en_core_web_sm",
            "medium" : "en_core_web_md",
            "large" : "en_core_web_lg",
            "trf" : "en_core_web_trf"
        }
    def execute_subprocess(self,args):
        process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        print(process.stdout.decode())

    def download_model(self,model_name):
        spacy_model_download(model_name)
        print(f"{model_name} downloaded")

    def read_data(self,data_path):
        if round(sum(self.split_ratio),10) != 1.0:
            raise ValueError("Sum of split_ratio should be 1")

        ratio_len = len(self.split_ratio)
        if ratio_len != 2:
            raise ValueError("split_ratio must sum to 1 consist of 2 values")

        with open(data_path, 'rb') as file:
            TRAINING_DATA = pickle.load(file)
            trainlogger.info(f"[{__name__}] ({FUNC_NAME()}) TRAINING_DATA : {TRAINING_DATA}")

            TRAINING_DATA = list(map(lambda x: (x[0], x[1]["entities"]), TRAINING_DATA))
            random.shuffle(TRAINING_DATA)

            input_len = len(TRAINING_DATA)

            train_idx = int(abs(input_len*self.split_ratio[0]))
            training_data = TRAINING_DATA[:train_idx]
            testing_data = TRAINING_DATA[train_idx:]

            if self.val_split:
                train_idx = int(len(training_data) *.8)
                training_data = TRAINING_DATA[:train_idx]
                validation_data = TRAINING_DATA[train_idx:]
                return training_data, validation_data, testing_data
            else:
                return training_data, testing_data

    def crete_docbin(self,data,data_folder,_type):
        # the DocBin will store the example documents
        db = DocBin()
        for text, annotations in data:
            doc = self.nlp(text)
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                if span is not None: # ADDDED THIS LINE AS WAS GETTING NONE
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        path = f"configs/{_type}.spacy"
        path = os.path.join(data_folder,path)
        # os.chmod(path, mode=0o777)
        db.to_disk(path)
        return path

    def create_base_config(self,data_path,cfg_path="./configs/config.cfg"):
        if self.resume:
            output_config = "base_config_resume.cfg"
        else:
            output_config = "base_config.cfg"
        # print(output_config)
        output_config = os.path.join(data_path,output_config)
        spacy_fill_config(output_file=Path(output_config),base_path=cfg_path)
        if self.resume:
            output_config = self.update_config_to_resume(output_config,self.model_save_path)
        spacy_fill_config(output_file=Path(output_config),base_path=cfg_path)
        return output_config

    def start_training(self,
                       cfg_path="config.cfg",
                       train_spacy_path="train.spacy",
                       dev_spacy_path="dev.spacy",
                       model_save_path= "ml_parser_api/libs/concentrix_ml_parser/model_trainer/trained_model/ner",):
        print("Model Training Started.....")
        overrides = {"paths.train": train_spacy_path, "paths.dev": dev_spacy_path}
        spacy_train(Path(cfg_path),
                    Path(model_save_path),
                    overrides=overrides,
                    use_gpu=-1)
        print("Model Training Completed.....")

    def setup_model(self,model_size):

        # Read Model
        # print(type(model_size),  model_size)
        model_name = self.model_size_dict.get(model_size)
        try:
            nlp = spacy.load(model_name)
        except:
            print(f"Downloading Model : {model_name}")
            self.download_model(model_name)
            nlp = spacy.load(model_name)
        print("Model Loaded")

        if self.setup_transformer:
            # Get the transformer component from the pipeline
            transformer = nlp.get_pipe("transformer")
            # Freeze the embeddings
            transformer.cfg['update_vectors'] = False
            print("transformer.is_trainable : ",transformer.is_trainable)

        print("All NLP pipelines: ",nlp.pipe_names)

        return nlp

    def update_config_to_resume(self,data_path,config_path,model_save_path):
        best_model_path = os.path.join(model_save_path,"model-best")
        if os.path.exists(best_model_path):
            print("BEST MODEL EXIST")
        new_config_path = os.path.join(data_path,"base_config_resume.cfg")
        if os.path.exists(config_path):
            print(f"{config_path} exists")
        base_config = Config().from_disk(config_path)
        base_config['components']['ner']['source'] = best_model_path
        base_config['components']['transformer']['source'] = best_model_path
        filled_config = Config().from_disk(config_path)
        filled_config.update(base_config)
        filled_config.to_disk(new_config_path)
        return new_config_path


    def ner_main(self, data_path, model_size,
                 split_ratio=(.8,.2), val_split=False, setup_transformer=False,
                 **kwargs) -> None:

        self.resume = kwargs.get("resume",False)
        self.setup_transformer = kwargs.get("setup_transformer",False)
        self.best_or_last = kwargs.get("best_or_last","best")
        model_save_path = kwargs.get('model_path')
        base_data_path = kwargs.get('base_data_path')
        base_cfg_path =  kwargs.get('base_cfg_path')
        os.chmod(model_save_path, mode=0o777)

        # Read Data
        self.split_ratio = split_ratio
        self.val_split = val_split
        training_data, testing_data = self.read_data(data_path)
        # print("Data Splitted")
        # Setup Model
        self.nlp = self.setup_model(model_size)
        # Generate DocBin
        train_spacy_path = self.crete_docbin(training_data,base_data_path,"train")
        dev_spacy_path = self.crete_docbin(testing_data,base_data_path,"dev")
        trainlogger.info(f"[{__name__}] ({FUNC_NAME()}) train & dev created")
        base_config = self.create_base_config(base_data_path,base_cfg_path)
        self.start_training(cfg_path=base_config,
            train_spacy_path=train_spacy_path,
            dev_spacy_path=dev_spacy_path,
            model_save_path=model_save_path)
