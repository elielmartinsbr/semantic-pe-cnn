def get_dataset_path():
    import os
    return os.path.join('VirusShareSant')

def get_precessed_path(part):
    import os
    get_dataset_path()
    return os.path.join(get_dataset_path(),'pe-parts',part)


def dataset_load(path=None,verbose=1,balanced=None):
    import numpy as np
    import pandas as pd
    import os
    from sklearn.utils.class_weight import compute_class_weight

    if path == None:
        path=get_dataset_path()

    print("\nLoading original malware metadata from VirusShareSant...")
    X_exp=np.load(os.path.join(path,"X_exp3.npy")) # file names, each line represent one malware
    y_exp=np.load(os.path.join(path,"y_exp3.npy")) # labels

    files = pd.DataFrame(X_exp,columns=['file'])
    labels = pd.DataFrame(y_exp,columns=['label'])

    if verbose:
        print('X_exp: ', X_exp.shape)
        print('y_exp: ', y_exp.shape)

    # malware family names
    malware_family_names=["Backdoor:Win32/Bifrose","Trojan:Win32/Vundo","Exploit:Win32/CplLnk.A","VirTool:Win32/Obfuscator",
       "Worm:Win32/Citeary","Backdoor:Win32/Cycbot","BrowserModifier:Win32/Zwangi","Rogue:Win32/Winwebsec",
       "Trojan:Win32/Koutodoor","TrojanDownloader:Win32/Troxen","PWS:Win32/OnLineGames","Backdoor:Win32/Ripini",
       "Backdoor:Win32/Rbot","Backdoor:Win32/Hupigon","Worm:Win32/Hybris","BrowserModifier:Win32/Istbar",
       "Trojan:Win32/Startpage"]

    # class index used
    classes_index=np.array([0,1,5,6,7,8,12,13,16])

    classes_names_used = []
    for i in classes_index:
        classes_names_used.append(malware_family_names[i])

    class_weights_computed = compute_class_weight(class_weight='balanced',
                                 classes=np.unique(y_exp),
                                 y=y_exp)
    norm = np.linalg.norm(class_weights_computed,ord=1)
    class_weights_computed = class_weights_computed/norm

    classes_weights = {}
    for i in range(len(np.unique(y_exp))):
        classes_weights[i] = class_weights_computed[i]

    if balanced == 'over':
        print("\nOver sampling dataset...")
        from imblearn.over_sampling import RandomOverSampler

        # define oversampling strategy
        oversample = RandomOverSampler(sampling_strategy='auto')

        X_over, y_over = oversample.fit_resample(files,labels)

        if verbose:
            print('X_over: ', X_over.shape)
            print('y_over: ', y_over.shape)

        df = pd.merge(X_over,y_over,left_index=True, right_index=True)

        return df, classes_names_used, classes_weights

    if balanced == 'under':
        print("\nUnder sampling...")
        from imblearn.under_sampling import RandomUnderSampler

        # define oversampling strategy
        undersample = RandomUnderSampler(sampling_strategy='auto')

        X_under, y_under = undersample.fit_resample(files,labels)

        df = pd.merge(X_under,y_under,left_index=True, right_index=True)

        return df, classes_names_used, classes_weights


    if balanced == None:
        print("\nNot balanced...")
        df = pd.merge(files,labels,left_index=True, right_index=True)
        return df, classes_names_used, classes_weights


def dataset_split_k_fold(dataset, verbose=0, mum_samples=0):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print('\nSplitting Data into Training, Validation, and Testing...')

    # Shuffle all data in dataframe
    df = dataset.sample(frac=1).reset_index(drop=True)

    if mum_samples > 0:
        print('\nmum_samples: ', mum_samples)
        df = df.sample(n=mum_samples).reset_index(drop=True)

    # Splitting the data into training, validation and test sets
    train_df = df.sample(frac=0.8)
    val_df = df.drop(train_df.index).sample(frac=0.5)
    test_df = df.drop(train_df.index).drop(val_df.index).sample(frac=1)

    if verbose:
        print('train:', train_df.shape)
        print(' test:', test_df.shape)
        print('  val:', val_df.shape)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Load a batch of files into memory
#
# Parameters:
# - pe_part: file part
# - pe_size: limits the amount of data loaded from disk
def get_pe_batch_numpy(dataset, files_path, pe_part, pe_size, verbose=1):

    from threading import Thread
    import logging
    import numpy as np
    import utils
    import os

    def get_pe_data_from_file(files_path, filename, pe_part_array, pe_part, pe_size, index):
        try:
            file_path = os.path.join(files_path,filename+'_'+pe_part+'.npy')
            pe_part_array[index, :] = np.load(file_path)[:pe_size]

        except:
            logging.error('Error load %s' % file_path)
        return True

    num_samples = len(dataset)
    files = dataset['file'].reset_index(drop=True)
    labels = dataset['label'].reset_index(drop=True).astype('uint8').to_numpy()

    pe_part_array = np.zeros((num_samples, pe_size), dtype=np.uint8)

    threads = []
    for i, file in enumerate(files):
        process = Thread(target=get_pe_data_from_file, args=[files_path, file, pe_part_array, pe_part, pe_size, i])
        process.start()
        threads.append(process)

    for process in threads:
        process.join()

    if verbose:
        utils.print_size_and_shape(pe_part_array,name=pe_part)

    return pe_part_array, labels



def load_data_to_memory(pe_part, pe_size, dataset=None, files_path=None,
                        verbose=0, mum_samples=0,balanced=None):
    import utils
    t0 = utils.print_start_time()


    if files_path == None:
        files_path = get_precessed_path(pe_part)
    if dataset == None:
        dataset, classes_names, classes_weights = dataset_load(balanced=balanced)
    train_df, val_df, test_df = dataset_split_k_fold(dataset,mum_samples=mum_samples,verbose=verbose)


    if verbose:
        print('\nLoad train data')
    X_train, y_train = get_pe_batch_numpy(train_df, files_path, pe_part, pe_size, verbose=verbose)

    if verbose:
        print('\nLoad val data')
    X_val, y_val = get_pe_batch_numpy(val_df, files_path, pe_part, pe_size, verbose=verbose)

    if verbose:
        print('\nLoad test data')
    X_test, y_test = get_pe_batch_numpy(test_df, files_path, pe_part, pe_size, verbose=verbose)


    if verbose:
        print('\nSamples data:')
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        print("X_val: ", X_val.shape)
        print("y_val: ", y_val.shape)
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        tx = utils.print_end_time(t0)

    return X_train, X_test, X_val, y_train, y_test, y_val


def load_data_to_memory_batch(pe_part, pe_size, dataset=None, files_path=None,
                        verbose=0, mum_samples=0,balanced=None):
    import utils
    t0 = utils.print_start_time()


    if files_path == None:
        files_path = get_precessed_path(pe_part)
    if dataset == None:
        X,y,dataset, malware_family_names, classes_index, classes_weights = dataset_load(balanced=balanced)


    # Partition the data and separate into training and test sets
    df = dataset.sample(frac=1).reset_index(drop=True)

    if mum_samples > 0:
        df = df.sample(mum_samples)

    if verbose:
        print('\nLoad test data')
    X_test, y_test = get_pe_batch_numpy(df, files_path, pe_part, pe_size, verbose=verbose)


    if verbose:
        print('\nData samples:')
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        tx = utils.print_end_time(t0)

    return X_test, y_test, classes_index, classes_weights
