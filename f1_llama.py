#from analysis_functions import *
import numpy as np
import pandas as pd
import json
import os
from os.path import join
from sklearn.metrics import classification_report

def report_gen(y_pred, y_true, report_name=None, out_loc='report_llama'):
    # given predicted and true labels, 
    # generate the overall results and pertype analysis with misclassification
    report = classification_report(y_true, y_pred, output_dict=True)

    df_report = pd.DataFrame(columns=['type', 'precision', 'recall','f1-score', 'support'])

    overall = {}
    for t in report:
        if t not in ['accuracy','macro avg','weighted avg']:
            report[t]['type']=t
            df_report= pd.concat([df_report, pd.DataFrame(report[t],index=[0])], ignore_index=True)
        else:
            overall[t] = report[t]


    # extract misclassification details
    dic = {}
    for t,p in zip(y_true, y_pred):
        if t not in dic:
            dic[t] = {'mis_to':{}, 'mis_from':{}}
        if p not in dic:
            dic[p] = {'mis_to':{}, 'mis_from':{}}
            
        if t!=p:
            dic[t]['mis_to'][p] = dic[t]['mis_to'].get(p, 0) + 1
            dic[p]['mis_from'][t] = dic[p]['mis_from'].get(t, 0) + 1

    def first_five(dic):
        return sorted(dic.items(), key=lambda x: x[1], reverse=True)[:5]

    df_report['mis_from_top5'] = df_report.apply(lambda x: first_five(dic[x['type']]['mis_from']),axis=1) # precision
    df_report['mis_to_top5'] = df_report.apply(lambda x: first_five(dic[x['type']]['mis_to']),axis=1) # recall

    # save results
    if report_name is not None:
        if not os.path.exists(out_loc):
            os.mkdir(out_loc)

        df_report.sort_values(['f1-score'], ascending=False).to_csv(join(out_loc,'results_per_type_{}.csv'.format(report_name)))

        with open(join(out_loc, 'overall_{}.json'.format(report_name)), 'w') as outfile:  
            json.dump(overall, outfile)

    return overall, df_report

if __name__ == "__main__":
    preds = np.load('all_pred_single.npy')
    trues = np.load('all_true_single.npy')
    #maxlen = min(preds.shape[0], trues.shape[0])
    #report_gen(preds[:maxlen], trues[:maxlen])
    overall, report = report_gen(preds,trues)
    with open("overall_single.json","w") as f:
        json.dump(overall,f)
    report.to_csv('report_single.csv', index=False)