# Returns a dataframe with the results of the aggregated test series

import argparse
import warnings

import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import time
from keras.models import load_model
import utils as utils


def load_data_results(pe_part,path_results='train_data_men',
    result_index=0, return_models=False, best=False):

    utils.set_tensoflow_limits()

    def calc_percent_preds(y_pred,y_true,classes):
        true_labels = classes[y_true]
        pred_labels = classes[np.argmax(y_pred, axis=1)]
        return (sum(pred_labels == true_labels)/len(true_labels)*100)

    classes=np.array([0,1,5,6,7,8,12,13,16])

    path = os.path.join(path_results,pe_part)
    name_list = os.listdir(path)
    full_list = [os.path.join(path,i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime, reverse=True)
    for k,v in enumerate(time_sorted_list):
        print(k,v)

    if result_index == None:
        result_path = time_sorted_list[0]
    else:
        result_path = time_sorted_list[result_index]
    print('result_path in use: ',result_path)


    path_pattern = os.path.join(result_path,'*_model_trained.hdf5')
    list_models = glob.glob(path_pattern)

    if return_models:
        return list_models, result_path

    if len(list_models) > 0:

        df = pd.DataFrame()

        for i in range(len(list_models)):
            index=i+1
            round_path = ''.join([result_path,'/',str(index),'_'])
            df_load = pd.read_csv(round_path+'history.csv')
            if best:
                df_load = df_load[df_load.val_sparse_categorical_accuracy == df_load.val_sparse_categorical_accuracy.max()]
                df_load = df_load[df_load.val_loss == df_load.val_loss.min()]
            else:
                df_load = df_load.tail(1)

            df = pd.concat([df,df_load],ignore_index=True)



            evaluate = np.load(round_path+'evaluate.npy')
            df.loc[i,'eval_loss'] = evaluate[0]
            df.loc[i,'eval_acc'] = evaluate[1]

            df.loc[i,'pred %'] = calc_percent_preds(
                np.load(round_path+'y_pred.npy'),
                np.load(round_path+'y_true.npy'),
                classes
            )

            df.loc[i,'duration'] = np.load(round_path+'duration.npy')
            df.loc[i,'time'] = np.load(round_path+'duration.npy')


            model = load_model(round_path+'model_trained.hdf5')
            params = model.count_params()
            imput_len = model.input.shape[1]

            df.loc[i,'params'] = params
            df.loc[i,'imput_len'] = imput_len

            df.loc[i,'round'] = index


        df.insert(0, 'params', df.pop('params'))
        df.insert(0, 'imput_len', df.pop('imput_len'))
        df.insert(0, 'round', df.pop('round'))

        df = df.rename(columns={"Unnamed: 0": "epoca",
                                    "sparse_categorical_accuracy": "acc",
                                   "val_sparse_categorical_accuracy": "val_acc"})

        df['acc'] = df['acc'].map(lambda a: a*100)
        df['val_acc'] = df['val_acc'].map(lambda a: a*100)
        df['eval_acc'] = df['eval_acc'].map(lambda a: a*100)

        df.loc['media'] = df.mean()

        def cal_duration(row):
            import time
            ty_res = time.gmtime(int(row['duration']))
            return time.strftime("%H:%M:%S",ty_res)

        df['duration'] = df.apply(cal_duration, axis=1)

        df.loc['media','round'] = 0
        df = df.astype(dtype= {"round":"int64","epoca":"int64","params":"int64","imput_len":"int64"})

        return df

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='SPEMC - Arguments')
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--pe_part', type=str, default='header')

# Usage:
# python results.py -h

if __name__ == '__main__':
    print('-'*80)
    print('Semantic PE Malware Classifier (SPEMC) - Results')
    print('-'*80)

    args = parser.parse_args()

    print('Results for part: ',args.pe_part)
    df = load_data_results(args.pe_part)

    print('-'*80)
    print(df)
