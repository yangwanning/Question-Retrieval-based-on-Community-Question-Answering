import matchzoo as mz
import os

class MVLSTMConfig():
    # preprocessor = mz.preprocessors.ANMMPreprocessor()
    optimizer = 'SGD'
    model = mz.models.MVLSTM
    generator_flag = 1
    name = 'mvlstm'

    num_dup = 1
    num_neg = 4
    shuffle = True

    ###### training config ######
    batch_size = 128
    epoch = 1000

    ##### save config #####
    parent_path = '/ssd2/wanning/matchzoo/saved_results'
    model_parent_path = '/ssd2/wanning/matchzoo/MVLSTM'
    model_save_path = os.path.join(model_parent_path,name)
    learning_rate = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
    # dropout_rate = [0.1, 0.2]
    num_layers = [1]
    parameter_set = []
    parameter_set1 = []
    # parameter_set = [dropout_rate,num_layers]
    for dr in dropout_rate:
        for nl in num_layers:
            parameter_set.append([dr,nl])

    for dr in dropout_rate:
        for nl in num_layers:
            for lr in learning_rate:
                parameter_set1.append([dr,nl])

    parameter_name = ['dropout_rate','num_layers']