from inrct.models.my_model import *

def get_model(model_cfg):

    if model_cfg['encoder_name'] == 'Default':
        model = MyModel_default(model_cfg)
    else:
        raise NotImplementedError
    
    return model