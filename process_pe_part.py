# Logic for parallel processing of PE files

import argparse
import warnings
import os
import shutil
import numpy as np
import multiprocessing
from functools import partial

import utils
import dataset
import pe


def pe_split_and_save(path_of_files, path_to_save, pe_part, pe_size, file):

    file_path = os.path.join(path_of_files, file + '.gz')

    h,c,d,f = pe.get_all_data_from_pe(file_path)

    file_to_save = os.path.join(path_to_save, file)

    def data_save_npy(data,pe_size,name):
        data = data[:pe_size]
        data = np.pad(data, (0, pe_size - max(0,len(data))), 'constant')
        np.save(''.join([file_to_save,'_',name,'.npy']), data)

    if pe_part == 'header':
        data_save_npy(h,pe_size,pe_part)
    if pe_part == 'code':
        data_save_npy(c,pe_size,pe_part)
    if pe_part == 'data':
        data_save_npy(d,pe_size,pe_part)
    if pe_part == 'file':
        data_save_npy(f,pe_size,pe_part)
    if pe_part == 'all':
        data_save_npy(h,pe_size,'header')
        data_save_npy(c,pe_size,'code')
        data_save_npy(d,pe_size,'data')
        data_save_npy(f,pe_size,'file')

def parallel_runs(data_list, path_of_files, path_to_save, pe_part, pe_size, max_processes):
    print('\nParallel processing files')
    pool = multiprocessing.Pool(processes=max_processes)

    partial_pe_split_save=partial(pe_split_and_save, path_of_files, path_to_save, pe_part, pe_size)
    pool.map(partial_pe_split_save, data_list)


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='SPEMC - Arguments')
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--pe_part', type=str, default='all')
parser.add_argument('--pe_size', type=int, default=4096)
parser.add_argument('--dataset_path', type=str, default='VirusShareSant')
parser.add_argument('--max_processes', type=int, default=10)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--cleanup', action="store_true")


# Usage:
# python process_pe_part.py -h

if __name__ == '__main__':
    __spec__ = None

    print('-'*80)
    print('Semantic PE Malware Classifier (SPEMC) - Process PE Files')
    print('-'*80)

    args = parser.parse_args()

    t0 = utils.print_start_time()

    path_to_save = os.path.join(args.dataset_path,'pe-parts',args.pe_part)
    path_of_files = os.path.join(args.dataset_path,'pe-malwares')
    os.makedirs(path_to_save, exist_ok=True)

    if args.cleanup:
        print('\nRemoving all the existing processed files')
        shutil.rmtree(path_to_save,)
        os.makedirs(path_to_save, exist_ok=True)

    dataset, classes_names, class_weights = dataset.dataset_load()


    parallel_runs(dataset['file'], path_of_files, path_to_save, args.pe_part, args.pe_size, args.max_processes)

    tx = utils.print_end_time(t0)
