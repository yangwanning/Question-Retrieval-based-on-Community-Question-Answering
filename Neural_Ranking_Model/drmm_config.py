import matchzoo as mz
import os

class DRMMConfig():
    # preprocessor = mz.preprocessors.DRMMPreprocessor()
    optimizer = 'SGD'
    model = mz.models.DRMM
    generator_flag = 1
    name = 'DRMM'

    num_dup = 1
    num_neg = 4
    shuffle = True

    ###### training config ######
    batch_size = 20
    epoch = 20

    ##### save config #####
    parent_path = '/ssd2/wanning/matchzoo/saved_model/drmm'
    save_path = os.path.join(parent_path,'')