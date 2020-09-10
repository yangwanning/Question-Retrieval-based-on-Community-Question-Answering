import matchzoo as mz
import os
from dssm_config import DSSMConfig
from anmm_config import ANMMConfig
from knrm_config import KNRMConfig
from cdssm_config import CDSSMConfig
from duet_config import DUETConfig
from arcii_config import ArcIIConfig
from mvlstm_config import MVLSTMConfig
import shutil
import pickle
import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os
import pandas as pd
import keras
import subprocess as sp
import re
import numpy as np

def model_delete(path):
    trash_dir = path
    try:
        shutil.rmtree(trash_dir)
    except OSError as e:
        print(f'Error: {trash_dir} : {e.strerror}')
    os.mkdir(trash_dir)


def eva(runfile_path,qrels_file):
    run_args = '/ssd/home/wanning/anserini/eval/trec_eval.9.0.4/trec_eval' + ' ' + '-m' + ' ' + 'all_trec' + ' ' + '-M' + ' ' + '1000' + ' ' +  qrels_file + ' ' + runfile_path
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()
    # print(out)
    MAP = re.findall(r'map\s+all.+\d+', out)[0].split('\t')[2].strip()
    P20 = re.findall(r'P_20\s+all.+\d+', out)[0].split('\t')[2].strip()
    NDCG20 = re.findall(r'ndcg_cut_20\s+all.+\d+', out)[0].split('\t')[2].strip()
    MRR = re.findall(r'recip_rank\s+all.+\d+', out)[0].split('\t')[2].strip()
    run_args = '/ssd/home/wanning/anserini/eval/trec_eval.9.0.4/trec_eval' + ' ' + '-m' + ' ' + 'P.1' + ' ' +  qrels_file + ' ' + runfile_path
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()
    P1 = re.findall(r'P_1\s+all.+\d+', out)[0].split('\t')[2].strip()
    return MAP,NDCG20,P20,MRR,P1

def rst_writter(bst_model,prediction,model_name,qrels,dataset_name):
    runfile_name = os.path.join('/ssd2/wanning/matchzoo/saved_results',dataset_name,'evaluation_files',model_name,bst_model)
    f1 = open(runfile_name,'w')
    data = pd.read_csv(os.path.join('/ssd/home/wanning/anaconda3/envs/my_tf/lib/python3.6/site-packages/matchzoo/datasets',dataset_name,'test.csv'), sep=',')
    qid = data.id_left.tolist()
    did = data.id_right.tolist()
    print(len(qid))
    print(len(did))
    print(len(prediction))
    if len(qid) == len(did) == len(prediction):
        for count,i in enumerate(qid):
            print('check')
            f1.write(i.strip().split('Q')[1] + '\t' + 'Q0' + '\t' + str(did[count]) + '\t' + '0' + '\t' + str(float(prediction[count])) + '\t' + 'indri' + '\n')
    f1.close()
    MAP, NDCG20, P20, MRR,P1 = eva(runfile_name,qrels)
    return float(MAP), float(NDCG20), float(P20), float(MRR),float(P1)
def setting():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def reset(counter):
    result[counter] = {}
    all_history[counter] = {}
    map = []
    mrr = []
    loss_recording = []
    map_recording = []
    mrr_recording = []
    ndcg = []
    p20 = []
    return map,mrr,ndcg,p20,loss_recording,map_recording,mrr_recording

class BasicModel(object):
    def __init__(self, config):
        self.config = config

    def mkdir(self):
        folder = os.path.exists(self.config.model_save_path)
        if not folder:
            os.makedirs(self.config.model_save_path)
            print("---  new folder...  ---")
            print("---  new folder...  ---")
        else:
            print("---  There is this folder!  ---")

    def parameter_get(self):
        return self.config.parameter_set,self.config.parameter_name
    def name(self):
        return self.config.name
    def get_path(self):
        return self.config.model_save_path,self.config.model_parent_path

    def model_delete(self):
        trash_dir = self.config.model_parent_path
        try:
            shutil.rmtree(trash_dir)
        except OSError as e:
            print(f'Error: {trash_dir} : {e.strerror}')
        os.mkdir(trash_dir)

    def auto_prepare(self,train_pack,valid_pack,test_pack):
        model = self.config.model()
        task = mz.tasks.Ranking(metrics=['mean_average_precision','mean_reciprocal_rank'])
        model.params['task'] = task
        model_ok, train_ok, preprocesor_ok = mz.auto.prepare(model=model, data_pack=train_pack)
        test_ok = preprocesor_ok.transform(test_pack, verbose=0)
        valid_ok = preprocesor_ok.transform(valid_pack, verbose=0)
        return model_ok, train_ok, test_ok, valid_ok
    def get_lr(self):
        return self.config.learning_rate
def load_data(data):
    ####### load data ########
    train_pack = data.load_data('train', task='ranking')
    valid_pack = data.load_data('dev', task='ranking')
    test_pack = data.load_data('test', task='ranking')
    return train_pack,valid_pack,test_pack


if __name__ == '__main__':
    #### create new session #####
    setting()
    dataset_name = 'faq'
    two_dataset = {'faq': mz.datasets.faq,
                   'robust': [mz.datasets.robust1, mz.datasets.robust2, mz.datasets.robust3, mz.datasets.robust4,
                              mz.datasets.robust5],'quora':mz.datasets.quora}
    train_data_pack, valid_data_pack, test_data_pack = load_data(mz.datasets.faq)
    # qrels = '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt'
    # qrels = '/ssd/wanning/dataset/NPRF/all_topic/qrels.robust2004.txt'
    qrels_file = '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt'
    qrels = '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt'
    # qrels = '/ssd2/wanning/Quora/quora_qrels'

    # configs = [MVLSTMConfig()]
    # configs = [ANMMConfig(), ArcIIConfig(), DUETConfig(),MVLSTMConfig()]
    configs = [ANMMConfig()]

    all_history = {}
    result = {}
    for config in configs:
        model = BasicModel(config)
        model_name = model.name()
        # parameter_set,parameter_name = model.parameter_get()

        map = []
        mrr = []
        ndcg = []
        p20 = []
        all_history[model_name] = {}
        result[model_name] = {}
        map, mrr, ndcg,p20,loss_recording, map_recording, mrr_recording = reset(iter)
        bst_score = 0
        for iter in range(5):
            model.model_delete()
            model.mkdir()
            ###### TRAIN ######
            model_ok, train_ok, test_ok, valid_ok = model.auto_prepare(train_data_pack, valid_data_pack, test_data_pack)
            callback = mz.engine.callbacks.EvaluateAllMetrics(model_ok, *valid_ok.unpack(), batch_size=128,
                                                              verbose=0,
                                                              model_save_path=model.get_path()[0])
            history = model_ok.fit(*train_ok.unpack(), batch_size=128, epochs=20, callbacks=[callback])

            ########## FINDING BST ONE ##########
            bst_model_map = max(history.history[list(history.history.keys())[1]])
            bst_model_index = history.history[list(history.history.keys())[1]].index(bst_model_map)
            name_index = bst_model_index + 1
            print('bst_model_index:{0}'.format(bst_model_index))
            bst_model_name = ''.join((model_name, str(name_index)))
            print('bst_model_name:{0}'.format(bst_model_name))
            ######### TESTING ########
            test_model = mz.engine.base_model.load_model(os.path.join(model.get_path()[1], bst_model_name))
            test_x, test_y = test_ok.unpack()
            prediction = test_model.predict(test_x)
            # print(prediction)
            ##### EVALUATING #####
            MAP, NDCG20, P20, MRR,P1 = rst_writter(('_'.join((str(iter),bst_model_name))), prediction, model_name, qrels,dataset_name)
            map.append(MAP)
            mrr.append(MRR)
            ndcg.append(NDCG20)
            p20.append(P1)
            ##### HISTORY RECORDING ######
            loss_recording.append(history.history[list(history.history.keys())[0]])
            map_recording.append(history.history[list(history.history.keys())[1]])
            mrr_recording.append(history.history[list(history.history.keys())[2]])
            if MAP > bst_score:
                bst_score = MAP
                bst_index = iter

        all_history[model_name]['loss'] = loss_recording
        all_history[model_name]['map'] = map_recording
        all_history[model_name]['mrr'] = mrr_recording

        result[model_name]['map'] = [np.mean(map),np.std(map)]
        result[model_name]['mrr'] = [np.mean(mrr),np.std(mrr)]
        result[model_name]['p1'] = [np.mean(p20), np.std(p20)]
        # result[model_name]['ndcg'] = ndcg
        # result[model_name]['p1'] = [np.mean(p20),np.std(p20)]
        result[model_name]['bst_score'] = bst_score
        result[model_name]['bst_index'] = bst_index
    # for key,item in result.items():
    #     print(key)
    #     print('{2}: map:{0} {1}'.format(np.mean(result[key]['map']),np.std(result[key]['map']),key))
    #     print('{2}: mrr:{0} {1}'.format(np.mean(result[key]['mrr']),np.std(result[key]['mrr']),key))
    #     # print('ndcg:{0} {1}'.format(np.mean(ndcg),np.std(ndcg)))
    #     # print('p:{0} {1}'.format(np.mean(p20),np.std(p20)))
    #     print('{1}: bst score:{0}'.format(result[key]['bst_score'],key))
    #     print('{1}: bst index:{0}'.format(result[key]['bst_index'],key))

    print(result)
    # pickle.dump(all_history,open(os.path.join('/ssd2/wanning/matchzoo/saved_result_quora',dataset_name,'history' ), "wb"))
    # pickle.dump(all_history,open(os.path.join('/ssd2/wanning/matchzoo/saved_result_quora',dataset_name,'result'), "wb"))
    # pickle.dump(all_history,
    #             open(os.path.join('/ssd2/wanning/matchzoo', dataset_name, 'history'), "wb"))
    # pickle.dump(all_history,
    #             open(os.path.join('/ssd2/wanning/matchzoo', dataset_name, 'result'), "wb"))








