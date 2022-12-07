# Recover a batch of data from archives

def get_pe_batch(ds, files_path, pe_part, pe_size, batch_size, batch, verbose=0):
    import dataset
    start = batch*batch_size

    X, y = dataset.get_pe_batch_numpy(
            ds[start:start+batch_size],
            files_path,
            pe_part,
            pe_size,
            verbose=verbose,
        )

    if verbose:
        print('\nDados lote %d %s:' % (batch, pe_part))
        print("X: ", X.shape)
        print("y: ", y.shape)

    return X,y


import tensorflow as tf
import numpy as np
class DataGeneratorSequence(tf.keras.utils.Sequence):

    def __init__(self, ds, files_path, pe_part, pe_size, batch_size, verbose, shuffle=False):
        # Initialization
        self.ds = ds
        self.files_path = files_path
        self.batch_size = batch_size
        self.pe_part = pe_part
        self.pe_size = pe_size
        self.verbose = verbose
        self.shuffle = shuffle

        self.datalen = len(ds)
        if self.shuffle:
            self.ds = ds.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        # data preprocessing
        x, y = get_pe_batch(self.ds, self.files_path, self.pe_part, self.pe_size,
                self.batch_size, index, verbose=self.verbose)
        return x, np.array(y)

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(self.datalen // self.batch_size)
