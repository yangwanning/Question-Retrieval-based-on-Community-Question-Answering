import matchzoo as mz
import os

class KNRMConfig():
    # preprocessor = mz.preprocessors.DSSMPreprocessor()
    optimizer = 'SGD'
    model = mz.models.KNRM
    generator_flag = 1
    name = 'knrm'

    num_dup = 1
    num_neg = 4
    shuffle = True

    ###### training config ######
    batch_size = 128
    epoch = 2000

    ##### save config #####
    parent_path = '/ssd2/wanning/matchzoo/saved_results'
    model_parent_path = '/ssd2/wanning/matchzoo/KNRM'
    model_save_path = os.path.join(model_parent_path,name)

    # test_result_path = os.path.join(parent_path,'test_result')


