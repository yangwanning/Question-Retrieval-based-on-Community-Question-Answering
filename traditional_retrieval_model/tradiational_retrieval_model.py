import subprocess as sp
import argparse
import multiprocessing
import sys
import json
import os, sys
import pickle
import numpy as np
import re

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def start_process():
    print('Starting', multiprocessing.current_process().name)

def case0(topic_now_path, para, runfile_path):
    # run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-fbDocs=' + str(5) + ' ' + '-fbOrigWeight=' + str(
    #     0.2) + ' ' + '-fbTerms=' + str(para) + ' ' + '>' + ' ' + runfile_path
    run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-baseline=tfidf,k1:' + str(para[2]) + ',b:'+ str(para[1]) +  ' ' + '>' + ' ' + runfile_path
    return run_args

def case1(topic_now_path, para, runfile_path):
    # run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-fbDocs=' + str(5) + ' ' + '-fbOrigWeight=' + str(
    #     0.2) + ' ' + '-fbTerms=' + str(para) + ' ' + '>' + ' ' + runfile_path
    run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-rule=method:d,mu:' + str(50) +  ' ' + '-fbDocs=' + str(para[1]) + ' ' + '-fbTerms=' + str(para[2])+' '+'>' + ' ' + runfile_path
    # print(run_args)
    return run_args
def case2(topic_now_path, para, runfile_path):
    # run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-fbDocs=' + str(5) + ' ' + '-fbOrigWeight=' + str(
    #     0.2) + ' ' + '-fbTerms=' + str(para) + ' ' + '>' + ' ' + runfile_path
    run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-rule=method:d,mu:' + str(260) +  ' ' + '-fbDocs=' + str(para[1]) + ' ' + '-fbTerms=' + str(para[2])+' '+'-fbOrigWeight='+str(para[3])+' '+'>' + ' ' + runfile_path
    # print(run_args)
    return run_args
def case3(topic_now_path, para, runfile_path):
    run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-rule=method:d,mu:' + str(para[1]) +  ' ' +'>' + ' ' + runfile_path
    # print(run_args)
    return run_args
def case4(topic_now_path, para, runfile_path):
    # run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-fbDocs=' + str(5) + ' ' + '-fbOrigWeight=' + str(
    #     0.2) + ' ' + '-fbTerms=' + str(para) + ' ' + '>' + ' ' + runfile_path
    run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-rule=method:d,mu:' + str(50) +  ' ' + '-fbDocs=' + str(10) + ' ' + '-fbTerms=' + str(para[1])+' '+'>' + ' ' + runfile_path
    # print(run_args)
    return run_args

def case5(topic_now_path, para, runfile_path):
    # run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-fbDocs=' + str(5) + ' ' + '-fbOrigWeight=' + str(
    #     0.2) + ' ' + '-fbTerms=' + str(para) + ' ' + '>' + ' ' + runfile_path
    run_args = './IndriRunQuery' + ' ' + topic_now_path + ' ' + '-rule=method:d,mu:' + str(50) +  ' ' + '-fbDocs=' + str(10) + ' ' + '-fbTerms=' + str(30) +' '+'-fbOrigWeight='+str(para[1])+' '+'>' + ' ' + runfile_path
    # print(run_args)
    return run_args


#### run query ####  call indri. parameters need to be optimize
def query_run(mu):
    parameter = parameter_set[mu]
    runfile_name = os.path.join(runfile_path, str(mu))
    switch = {
        'bk': case0,
        'term': case4,
        'term_doc': case1,
        'term_doc_weight':case2,
        'mu':case3,
        'only_weight':case5,
        # 'bk_doc': case1,
        # 'bkdoc_term': case2,
        # 'bk_weight': case3,
    }
    run_args = switch[parameter_set[mu][0]](topic_path, parameter, runfile_name)
    # print(run_args)
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()

def eva(mu):
    runfile_name = os.path.join(runfile_path, str(mu))
    run_args = '/ssd/home/wanning/anserini/eval/trec_eval.9.0.4/trec_eval' + ' ' + '-m' + ' ' + 'all_trec' + ' ' + qrels_path + ' ' + runfile_name
    # print(run_args)
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()
    MAP = float(re.findall(r'map\s+all.+\d+', out)[0].split('\t')[2].strip())
    P20 = float(re.findall(r'P_5\s+all.+\d+', out)[0].split('\t')[2].strip())
    NDCG20 = float(re.findall(r'ndcg\s+all.+\d+', out)[0].split('\t')[2].strip())
    MRR = float(re.findall(r'recip_rank\s+all.+\d+', out)[0].split('\t')[2].strip())
    # print('MRR:{0}'.format(MRR))
    return MAP,NDCG20,P20,MRR

if __name__ == '__main__':
        ####### train+dev #######
    title = 'term_doc_weight'
    bk_flag = 0
    term_flag = 0
    term_doc_flag = 0
    weight_flag = 1
    mu_flag = 0
    only_weight_flag = 0
    # parent_path = '/ssd2/wanning/Indri/runfile/Quora/BM25'
    parent_path = '/ssd2/wanning/Indri/runfile/FAQ/new_ql'
        # topic_path = '/ssd2/wanning/Quora/train_dev_topic'
    topic_path = '/ssd/wanning/dataset/FAQ/train_new_topic'
    runfile_path = os.path.join(parent_path, title)
    qrels_path = '/ssd/wanning/dataset/FAQ/qrels_file/faq_qrels_train_dev.txt'

    if only_weight_flag == 1:
        parameter_set = {}
        count = 0
        for k in list(np.arange(0.0, 1.1, 0.1)):
            if count <= 300000:
                parameter_set[count] = [title, k]
                count += 1

    if mu_flag == 1:
        parameter_set = {}
        count = 0
        for i in [1,5,10,25,50,75,100,250,500,750,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]:
            if count <= 300000:
                parameter_set[count] = ['mu', i]
                count += 1
    if term_flag == 1:
        parameter_set = {}
        count = 0
        for i in list(range(5, 500, 5)):
            if count <= 300000:
                parameter_set[count] = ['term',i]
                count += 1
    if weight_flag == 1:
        parameter_set = {}
        count = 0
        for i in list(range(5, 110, 5)):
            for j in list(range(1, 11, 1)):
                for k in list(np.arange(0.1, 1.0, 0.1)):
                    if count <= 300000:
                        parameter_set[count] = ['term_doc_weight', j, i,round(k, 2)]
                        count += 1
    if term_doc_flag == 1:
        parameter_set = {}
        count = 0
        #### term ####
        for i in list(range(5,50,1)):
            #### doc ####
            for j in list(range(5,11,1)):
                if count <=300000:
                    parameter_set[count] = ['term_doc', j,i]
                    count += 1

            ######### version2 ##########

    # if weight_flag == 1:
    #         parameter_set = {}
    #         count = 0
    #         for i in list(range(5, 25, 1)):
    #             for j in list(range(1, 11, 1)):
    #                 for k in list(np.arange(0.0, 1.1, 0.1)):
    #                     if count <= 300000:
    #                         parameter_set[count] = ['term_doc_weight', j, i, round(k, 2)]
    #                         count += 1
    if bk_flag == 1:
        parameter_set = {}
        count = 0
        for i in list(np.arange(0.1, 1.0, 0.05)):
            for j in list(np.arange(0.1, 4.0, 0.1)):
                if count <= 1000000:
                    parameter_set[count] = ['bk',round(i, 2),round(j, 2)]
                    count += 1
    counter = 0

    ####### Generate runfile #########
    for i in chunks(range(0, count), 200):
        pool_size = 20
        pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
        pool_outputs = pool.map_async(query_run, list(i))
        pool.close()  # no more tasks
        pool.join()  # wrap up current tasks

        print('finish one iteration {0}'.format(counter))
        counter += 1

    ######## Evaluation #######
    all_result = []
    counter = 0

    for i in chunks(range(0, count), 200):
        pool_size = 20
        pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
        pool_outputs = pool.map_async(eva, list(i))
        pool.close()  # no more tasks
        pool.join()  # wrap up current tasks


        print('finish one iteration for evaluation {0}'.format(counter))
        counter += 1
        all_result = all_result + pool_outputs.get()

    bst_score = 0
    for c,p in enumerate(all_result):
        if bst_score < p[0]:
            bst_score = p[0]
            bst_map = p[0]
            bst_ndcg = p[1]
            bst_p = p[2]
            bst_mrr = p[3]
            bst_index = c
    # print(all_result)
    print('map:{0}; ndcg:{1}; p:{2}; mrr:{3}; index:{4}'.format(bst_map,bst_ndcg,bst_p,bst_mrr,bst_index))
    print('para:{0}'.format(parameter_set[bst_index]))
    pickle.dump(all_result,open(os.path.join(parent_path,'result','_'.join((title,'result','new'))),"wb"))
    pickle.dump(parameter_set,open(os.path.join(parent_path,'result','_'.join((title,'para','new'))),"wb"))