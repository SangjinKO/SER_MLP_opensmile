import os
import Opensmile_Feature #Feature Extraction
from ML_Model import SVM_Model, MLP_Model # SVM, MLP - Train & Evaluate
from Config import Config #File Path

print ("TEST-MLP using opensmile for RAVEE")
print ("FEATURE: IS10_paraling")


def Train(save_model_name: str):
    Config.save_model_name = save_model_name
    x_train, x_test, y_train, y_test = Opensmile_Feature.get_data(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_OPENSMILE, train=True)
    model = MLP_Model()
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)

    return model


## Trainig & Validating
Train("MLP_LIB")

