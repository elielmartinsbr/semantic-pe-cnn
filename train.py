# Logic for processing, training, testing, and validating CNN models with the semantic parts

import argparse
import warnings
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import json
import subprocess

import utils as utils
import dataset as dataset
import batch as batch
import models as models
from tensorflow.keras import mixed_precision

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='SPEMC - Arguments')
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--rounds', type=int, default=1)

# Hiper parameters
parser.add_argument('--cnn_kernel', type=int, default=100)
parser.add_argument('--cnn_stride', type=int, default=50)
parser.add_argument('--cnn_filter', type=int, default=128)
parser.add_argument('--cnn_dense', type=int, default=64)
parser.add_argument('--cnn_decay_steps', type=int, default=30)
parser.add_argument('--cnn_decay_rate', type=float, default=1)

# Batch
parser.add_argument('--batch_size', type=int, default=32)

# Dataset
parser.add_argument('--dataset_path', type=str, default='VirusShareSant')
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--pe_size', type=int, default=5000)
parser.add_argument('--pe_increment', type=int, default=0)
parser.add_argument('--pe_part', type=str, default='header')
parser.add_argument('--pe_balanced', type=str, default='over')

parser.add_argument('--train_memory', action="store_true")
parser.add_argument('--train_generator', action="store_true")
parser.add_argument('--train_incremental', action="store_true")


if __name__ == '__main__':
    print('-'*80)
    print('Semantic PE Malware Classifier (SPEMC) - Training')
    print('-'*80)

    utils.set_tensoflow_limits()

    # https://www.tensorflow.org/guide/mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    args = parser.parse_args()


    if args.train_memory and args.train_generator:
        sys.exit('\nSelect only one training option.')

    files_path = os.path.join(args.dataset_path,'pe-parts',args.pe_part)
    data_time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


    if args.train_memory:
        train_results_path='train_data_men/'
    else:
        train_results_path='train_data_gen/'

    path_results_dir=os.path.join(train_results_path,args.pe_part,data_time_string)
    os.makedirs(path_results_dir, exist_ok=True)

    args_file = os.path.join(path_results_dir,'commandline_args.json')
    with open(args_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    start0 = utils.print_start_time(desc="Start the training")

    max_len = args.pe_size
    for i in range(args.rounds):
        index=i+1
        round_path = ''.join([path_results_dir,'/',str(index),'_'])

        if args.train_incremental and i > 0:
            max_len = max_len + args.pe_increment
            print('\nIncrement max_len: %d' % max_len)



        start1 = utils.print_start_time(desc="Start Round %d" % index)

        model = models.malconv(maxlen=max_len,
                                kernel=args.cnn_kernel,
                                stride=args.cnn_stride,
                                filters=args.cnn_filter,
                                dense=args.cnn_dense
                                )
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                            initial_learning_rate=0.01,
                            decay_steps=args.cnn_decay_steps,
                            decay_rate=args.cnn_decay_rate,
                            staircase=False
                        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                 patience=10,
                 verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                monitor='val_sparse_categorical_accuracy',
                filepath=round_path+'model-best-weights.hdf5',
                save_best_only=True)
        ]

        if args.train_generator:
            print('\nLoad data generator...')
            dataset_features, classes_names, class_weights = dataset.dataset_load(balanced=args.pe_balanced)
            train_df, val_df, test_df = dataset.dataset_split_k_fold(dataset_features, mum_samples=args.num_samples)
            y_test = test_df['label'].reset_index(drop=True).astype('uint8').to_numpy()

            def create_generator(ds,shuffle):
                return batch.DataGeneratorSequence(
                                ds,
                                files_path,
                                args.pe_part,
                                max_len,
                                args.batch_size,
                                verbose=args.verbose,
                                shuffle=shuffle
                            )

            training_generator = create_generator(train_df,True)
            validation_generator = create_generator(val_df,True)
            test_generator = create_generator(test_df,False)

            print('\nModel fit generator...')
            history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    workers=10,
                    use_multiprocessing=True,
                    epochs=1000,
                    callbacks=callbacks,
                    # class_weight=class_weights
            )
            print('\nModel evaluate generator...')
            eval_loss, eval_acc = model.evaluate(test_generator, verbose=args.verbose, return_dict=False)

            print('\nModel predict generator...')
            y_pred = model.predict(test_generator)


        if args.train_memory:
            print('\nLoad data to memory...')
            X_train, X_test, X_val, y_train, y_test, y_val = dataset.load_data_to_memory(
                                                                    files_path=files_path,
                                                                    pe_size=max_len,
                                                                    pe_part=args.pe_part,
                                                                    verbose=args.verbose,
                                                                    mum_samples=args.num_samples,
                                                                    balanced=args.pe_balanced)


            print('\Model fit...')
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val,y_val),
                epochs=1000,
                batch_size=args.batch_size,
                verbose=args.verbose,
                callbacks=callbacks,
            )
            print('\nModel evaluate...')
            eval_loss, eval_acc = model.evaluate(X_test, y_test, verbose=args.verbose, return_dict=False)

            print('\nModel predict...')
            y_pred = model.predict(X_test)


        print('\nStore trains and tests results...')

        pd.DataFrame(history.history).to_csv(round_path+'history.csv')
        np.save(round_path+'evaluate',[eval_loss, eval_acc])

        np.save(round_path+'y_pred', y_pred)
        np.save(round_path+'y_true', y_test)

        duration = utils.calc_time_durantion_in_seconds(start1)
        np.save(round_path+'duration', duration)

        model.save(os.path.join(round_path+'model_trained.hdf5'),save_format='h5')

        utils.print_end_time(start1,desc="End round %d" % index)

        print('-'*80)

    # round ends
    utils.print_end_time(start0,desc="End training")
