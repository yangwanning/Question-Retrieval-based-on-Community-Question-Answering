import os
import subprocess as sp
import sys
import re
import pickle
import numpy as np

def writter(output,write_output,qrels):
    fw = open(write_output,'w')
    with open(output, 'r') as f:
        for count, line in enumerate(f):
            items = line.split(' ')
            items[2] = items[2].split('=')[1]
            if qrels == '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt':
                items[0] = items[0].split('Q')[1]
            fw.write(' '.join((items)))
    fw.close()


    map,ndcg,p,mrr,p1 = eva(write_output,qrels)
    return map,ndcg,p,mrr,p1

def eva(runfile_path,qrels_file):
    run_args = '/ssd/home/wanning/anserini/eval/trec_eval.9.0.4/trec_eval' + ' ' + '-m' + ' ' + 'all_trec' + ' ' + '-M' + ' ' + '1000' + ' ' +  qrels_file + ' ' + runfile_path
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()
    MAP = re.findall(r'map\s+all.+\d+', out)[0].split('\t')[2].strip()
    P20 = re.findall(r'P_20\s+all.+\d+', out)[0].split('\t')[2].strip()
    NDCG20 = re.findall(r'ndcg_cut_20\s+all.+\d+', out)[0].split('\t')[2].strip()
    MRR = re.findall(r'recip_rank\s+all.+\d+', out)[0].split('\t')[2].strip()
    run_args = '/ssd/home/wanning/anserini/eval/trec_eval.9.0.4/trec_eval' + ' ' + '-m' + ' ' + 'P.1' + ' ' + qrels_file + ' ' + runfile_path
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()
    P1 = re.findall(r'P_1\s+all.+\d+', out)[0].split('\t')[2].strip()
    return MAP,NDCG20,P20,MRR,P1

if __name__ == '__main__':
    cv_flag = 0
    if cv_flag == 1:
        parent_path = '/ssd/home/wanning/indri-5.13/L2R-features/Robust_cv'
        folds = os.listdir(parent_path)
        qrels = '/ssd/wanning/dataset/NPRF/all_topic/qrels.robust2004.txt'
        output_path = '/ssd2/wanning/L2R/runfile/robust'
        runfile_path = '/ssd2/wanning/L2R/trec_runfile/robust'
    else:
        parent_path = '/ssd/home/wanning/indri-5.13/L2R-features'
        fold = 'quora'
        # fold = 'faq_new'
        # qrels = '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt'
        qrels = '/ssd2/wanning/Quora/quora_qrels'
        # qrels = '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt'
        output_path = '/ssd2/wanning/L2R/runfile/'
        runfile_path = '/ssd2/wanning/L2R/trec_runfile'
    all_result = {}
    final_result = {}
    # for ranker in [0,1,2,3,4,6,7,8]:
    # for ranker in [0,1,2,3,4,6,7,8]:
    for ranker in [3,4,6,8]:
    ###### save all results #####
        all_result[ranker] = {}
        # for feature in ['all', 'querydepend']:
        for feature in ['7queryfeature']:
            all_result[ranker][feature] = {}
            map_now = []
            ndcg_now = []
            p_now = []
            mrr_now = []
            for i in range(3):
                train = os.path.join(parent_path,fold,'.'.join(('-'.join(('outfile','train',feature)),'rl')))
                # print('train:{0}'.format(train))
                dev = os.path.join(parent_path,fold,'.'.join(('-'.join(('outfile','dev',feature)),'rl')))
                test = os.path.join(parent_path,fold,'.'.join(('-'.join(('outfile','test',feature)),'rl')))
                save_model = os.path.join('/ssd2/wanning/L2R',fold,'_'.join((str(ranker),fold,feature)))
                # print('path:{0}'.format(save_model))
                run_args = 'java' + ' ' + '-jar'+ ' ' +'RankLib-2.12.jar'+ ' ' +'-train'+ ' ' +train+ ' ' +'-validate'+ ' ' +dev+ ' ' +'-test'+ ' ' +test+ ' ' +'-ranker'+ ' ' +str(ranker)+ ' ' +'-metric2t'+ ' ' +'MAP'+ ' ' +'-save'+ ' ' +save_model
                # print(run_args)
                s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
                (out, err) = s.communicate()

                ######## load saved model and write into trec format #######
                output = os.path.join(output_path,fold,'_'.join((str(ranker),fold,feature)))
                args = 'java' + ' ' + '-jar'+ ' ' +'RankLib-2.12.jar'+ ' ' + '-load' + ' ' + save_model + ' ' + '-rank' + ' ' + test + ' ' + '-indri' + ' ' + output
                s1 = sp.Popen(args, stdout=sp.PIPE, shell=True, encoding='utf-8')
                (out, err) = s1.communicate()
                map,ndcg,p,mrr,p1 = writter(output,os.path.join(runfile_path,fold,'_'.join((str(ranker),fold,feature))),qrels)
                ######## recording #######
                map_now.append(float(map))
                ndcg_now.append(float(ndcg))
                p_now.append(float(p1))
                mrr_now.append(float(mrr))
                print(map)
                print('FINISH RANKER:{0}, FEATURE:{1}, ITERATION:{2}'.format(ranker,feature,i))
            ##### final result #####
            all_result[ranker][feature]['map'] = [np.mean(map_now),np.std(map_now)]
            all_result[ranker][feature]['mrr'] = [np.mean(mrr_now),np.std(mrr_now)]
            all_result[ranker][feature]['p'] = [np.mean(p_now),np.std(p_now)]

    # pickle.dump(all_result,open(os.path.join('/ssd2/wanning/L2R/results', fold), "wb"))

    print('all result:{0}'.format(all_result))

