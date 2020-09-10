import os
import subprocess as sp
import sys
import re
import pickle

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


    map,ndcg,p,mrr = eva(write_output,qrels)
    return map,ndcg,p,mrr

def eva(runfile_path,qrels_file):
    run_args = '/ssd/home/wanning/anserini/eval/trec_eval.9.0.4/trec_eval' + ' ' + '-m' + ' ' + 'all_trec' + ' ' + '-M' + ' ' + '1000' + ' ' +  qrels_file + ' ' + runfile_path
    s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
    (out, err) = s.communicate()
    MAP = re.findall(r'map\s+all.+\d+', out)[0].split('\t')[2].strip()
    P20 = re.findall(r'P_20\s+all.+\d+', out)[0].split('\t')[2].strip()
    NDCG20 = re.findall(r'ndcg_cut_20\s+all.+\d+', out)[0].split('\t')[2].strip()
    MRR = re.findall(r'recip_rank\s+all.+\d+', out)[0].split('\t')[2].strip()
    return MAP,P20,NDCG20,MRR

if __name__ == '__main__':
    cv_flag = 1
    if cv_flag == 1:
        parent_path = '/ssd/home/wanning/indri-5.13/L2R-features/Robust_cv'
        folds = os.listdir(parent_path)
        qrels = '/ssd/wanning/dataset/NPRF/all_topic/qrels.robust2004.txt'
        output_path = '/ssd2/wanning/L2R/runfile/robust'
        runfile_path = '/ssd2/wanning/L2R/trec_runfile/robust'
    else:
        parent_path = '/ssd/home/wanning/indri-5.13/L2R-features'
        folds = ['faq']
        qrels = '/ssd/home/wanning/anserini/FAQ/qrels_file_test.txt'
        qrels = '/ssd2/wanning/Quora/quora_qrels'
        output_path = '/ssd2/wanning/L2R/runfile/'
        runfile_path = '/ssd2/wanning/L2R/trec_runfile'

    all_result = {}
    final_result = {}


    # for ranker in [0,1,2,3,4,6,7,8]:
    for ranker in [0,1]:
    ###### save all results #####
        all_result[ranker] = {}
        all_result[ranker]['map'] = {}
        all_result[ranker]['ndcg'] = {}
        all_result[ranker]['p'] = {}
        all_result[ranker]['mrr'] = {}
    ##### display overall results #####
        final_result[ranker] = {}
        average_map = []
        average_ndcg = []
        average_p = []
        average_mrr = []
        for i in range(2):
            map_now = []
            ndcg_now = []
            p_now = []
            mrr_now = []
            for fold in folds:
                train = os.path.join(parent_path,fold,'train')
                # print('train:{0}'.format(train))
                dev = os.path.join(parent_path,fold,'dev')
                test = os.path.join(parent_path,fold,'test')
                save_model = os.path.join('/ssd2/wanning/L2R',fold,'_'.join((str(ranker),fold)))
                # print('path:{0}'.format(save_model))
                run_args = 'java' + ' ' + '-jar'+ ' ' +'RankLib-2.12.jar'+ ' ' +'-train'+ ' ' +train+ ' ' +'-validate'+ ' ' +dev+ ' ' +'-test'+ ' ' +test+ ' ' +'-ranker'+ ' ' +str(ranker)+ ' ' +'-metric2t'+ ' ' +'MAP'+ ' ' +'-save'+ ' ' +save_model
                # print(run_args)
                s = sp.Popen(run_args, stdout=sp.PIPE, shell=True, encoding='utf-8')
                (out, err) = s.communicate()

                ######## load saved model and write into trec format #######
                output = os.path.join(output_path,'_'.join((str(ranker),fold)))
                args = 'java' + ' ' + '-jar'+ ' ' +'RankLib-2.12.jar'+ ' ' + '-load' + ' ' + save_model + ' ' + '-rank' + ' ' + test + ' ' + '-indri' + ' ' + output
                s1 = sp.Popen(args, stdout=sp.PIPE, shell=True, encoding='utf-8')
                (out, err) = s1.communicate()
                map,ndcg,p,mrr = writter(output,os.path.join(runfile_path,'_'.join((str(ranker),fold))),qrels)
                ######## recording #######
                map_now.append(float(map))
                ndcg_now.append(float(ndcg))
                p_now.append(float(p))
                mrr_now.append(float(mrr))
                print('FINISH FOLD:{0}'.format(fold))
            ##### final result #####
            all_result[ranker]['map'][i] = map_now
            all_result[ranker]['ndcg'][i] = ndcg_now
            all_result[ranker]['p'][i] = p_now
            all_result[ranker]['mrr'][i] = mrr_now
            average_map.append(sum(map_now)/len(map_now))
            average_ndcg.append(sum(ndcg_now)/len(ndcg_now))
            average_p.append(sum(p_now)/len(p_now))
            average_mrr.append(sum(mrr_now)/len(mrr_now))
            print('FINISH ITERATION:{0}'.format(i))

        final_result[ranker]['map'] = sum(average_map)/len(average_map)
        final_result[ranker]['ndcg'] = sum(average_ndcg)/len(average_ndcg)
        final_result[ranker]['p'] = sum(average_p)/len(average_p)
        final_result[ranker]['mrr'] = sum(average_mrr) / len(average_mrr)
        print('FINISH RANKER:{0}'.format(ranker))

    pickle.dump(all_result,open(os.path.join('/ssd2/wanning/L2R/results', 'result_robust_10_01'), "wb"))

    print('final result:{0}'.format(final_result))

