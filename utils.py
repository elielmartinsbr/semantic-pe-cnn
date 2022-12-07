def set_tensoflow_limits():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def calc_percent_preds(y_pred, y_true, classes):
    import numpy as np

    true_labels = classes[y_true]
    pred_labels = classes[np.argmax(y_pred, axis=1)]
    return (sum(pred_labels == true_labels)/len(true_labels)*100)

def print_start_time(desc='Time Start'):
    from time import time
    from datetime import datetime
    start = time()
    now = datetime.now()
    print('\n%s: %s' % (desc,now))
    return start

def print_end_time(start,desc='Time End'):
    from time import time
    from datetime import timedelta
    from datetime import datetime

    tt = time()-start
    duration=timedelta(seconds=tt)

    print('\n%s: %s - %s' % (desc,datetime.now(),duration) )

    return duration

def calc_time_durantion_in_seconds(start):
    from time import time
    seconds = time()-start
    return seconds

def calc_readable(size,precision=2):
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 #increment the index of the suffix
        size = size/1024.0 #apply the division
    return "%.*f%s"%(precision,size,suffixes[suffixIndex])

def print_size_and_shape(array,name='',return_values=False):

    size = calc_readable(array.nbytes)
    shape = array.shape
    if return_values:
        return size,shape

    print(name, "- ", "Size: ", size, ", Shape: ",shape)
