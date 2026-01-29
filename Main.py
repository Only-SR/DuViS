import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, vgae_encoder, vgae_decoder, vgae
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict , l2_reg_loss , Metric
import os
from copy import deepcopy
import scipy.sparse as sp
import random
import importlib
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def import_model(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        print(f"Error importing {class_name} from {module_name}: {e}")
        return None

def main(model_name):
    handler = DataHandler()
    handler.LoadData()
    log('loading data:')
    if model_name == 'mine':
        ModelClass = import_model('models.model-mine', 'Coach',)
        model = ModelClass(handler)  
        model.run()
    else:
        print(f"Model {model_name} not found.")

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

if __name__ == '__main__':
    with torch.cuda.device(0):
        logger.saveDefault = True
        seed_it(args.seed)
        log('Start')
        name = args.model_name
        main(name)