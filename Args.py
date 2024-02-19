import os
import torch

class Args():
    SEED = 286338867

    # DEVICE = torch.device("cpu") 
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SURROGATE_TRAINING_SIZE = 1000
    FIDELITY_EVAL_SIZE = 100

    DIR_DATASET = "DataSet"
    DIR_MODEL = "Model"
    DIR_RESULT = "Result"

    URL_ADE20K = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" # http://data.csail.mit.edu/places/ADEchallenge/release_test.zip
    URL_PLACES_LABEL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
    URL_XCEPTION_ADE = "http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz"
    URL_RESNET_PLACES = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'

    NAME_ADE = "ADEChallengeData2016"
    NAME_PLACES_LABEL = "categories_places365.txt"
    NAME_XCEPTION_ADE = "deeplabv3_xception_ade20k_train_2018_05_29.tar.gz"
    NAME_RESNET_PLACES = "resnet50_places365.pth.tar"

    '''visualization'''
    ALPHA = 0.3
    DPI = 1200

    def get_path(self, name: str) -> str:
        if name == "ADE":
            return os.path.join(self.DIR_DATASET, self.NAME_ADE)
        elif name == "XCEPTION_ADE":
            return os.path.join(self.DIR_MODEL, self.NAME_XCEPTION_ADE)
        elif name == "RESNET_PLACES":
            return os.path.join(self.DIR_MODEL, self.NAME_RESNET_PLACES)
        else: 
            raise ValueError("Invalid name!")