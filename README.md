# Semantic PE Malware Classifier (SPEMC)

Repository of code used to generate the search results for the article "Semantic malware classification using convolutional neural networks".

The repository does not contain malware samples.

The instructions and command suggestions are examples of use for generating results compatible with those presented in the article.

Comments:
* Some code snippets may not have had the correct indication of the author, please let me know if I used codes, figures, graphics and did not cite the author correctly;
* Some comments, variable names and methods may be written in Portuguese or Spanish;
* GPU usage is required for full testing, if one is not available some warnings may be displayed;

## Environment Preparation

Install Python (https://www.python.org/), pip (https://pypi.org/project/pip/) e Conda (https://www.anaconda.com/products/distribution).

```
wget -c https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```

Create a python virtual environment with Anaconda for the tests:

```
conda create --name semantic-pe-cnn python=3.9 anaconda -y

conda activate semantic-pe-cnn

conda install -c anaconda py-lief==0.10.1 -y
conda install -c conda-forge imbalanced-learn -y

pip install tensorflow==2.9.0 keras==2.9.0 livelossplot

```

## Malware Dataset

Malware samples were provided by Ricardo Sant'Ana (https://github.com/ricksant2003/MalwareDatasetVirusShareSant).

To generate the sample data you need to request the binary files with the malware samples from the author of `VirusShareSant`, and unpack the archive contents into the directory "VirusShareSant/pe-malwares".

Then run the following commands to separate the files into semantic parts according to byte amounts used in the article's proofs:

```
python process_pe_part.py --num_samples=0 --pe_part=header --cleanup

python process_pe_part.py --pe_part=code --pe_size=100000 --num_samples=0 --cleanup

python process_pe_part.py --pe_part=data --pe_size=100000 --num_samples=0 --cleanup

python process_pe_part.py --pe_part=file --pe_size=250000 --num_samples=0 --cleanup
```

The results will be stored in the folder `VirusShareSant/pe-parts/{header,code,data,file}`.


## Training

To train the CNN models, test, evaluate and collect the results, the following commands must be executed:

```
python train.py --train_memory --verbose=1 --rounds=6 --pe_part=header --pe_size=4096 --num_samples=0

python train.py --train_memory --verbose=1 --rounds=6 --pe_part=file --pe_size=5000 --pe_increment=5000 --num_samples=0

python train.py --train_memory --verbose=1 --rounds=6 --pe_part=file --pe_size=5000 --pe_increment=5000 --num_samples=0


python train.py --train_memory --verbose=1 --rounds=6 --pe_part=file --pe_size=5000 --pe_increment=5000 --num_samples=0
```


## Results

To view the results run the commands below:

```
python results.py --pe_part=header

python results.py --pe_part=code

python results.py --pe_part=data

python results.py --pe_part=file

```


## Example

Just for the purpose of a more complete example, the following is a list of executed commands and result previews of training examples with 100 samples of the semantic part headers. The tests and data provided here do not represent any results used in the article.

Extract semantic header part from PE files:

```
(semantic-pe-cnn) $ python process_pe_part.py --num_samples=0 --pe_part=header --cleanup
--------------------------------------------------------------------------------
Semantic PE Malware Classifier (SPEMC) - Process PE Files
--------------------------------------------------------------------------------

Time Start: 2022-12-06 19:47:57.014530

Removing all the existing processed files

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Not balanced...

Parallel processing files

Time End: 2022-12-06 20:01:15.430244 - 0:13:18.398213
```

Training:

```
(semantic-pe-cnn) $ python train.py --train_memory --verbose=1 --rounds=6 --pe_part=header --pe_size=4096 --num_samples=100
--------------------------------------------------------------------------------
Semantic PE Malware Classifier (SPEMC) - Training
--------------------------------------------------------------------------------

Start the training: 2022-12-06 20:29:48.264245

Start Round 1: 2022-12-06 20:29:48.264315

Load data to memory...

Time Start: 2022-12-06 20:29:52.302580

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Over sampling dataset...
X_over:  (61146, 1)
y_over:  (61146, 1)

Splitting Data into Training, Validation, and Testing...

mum_samples:  100
train: (80, 2)
 test: (10, 2)
  val: (10, 2)

Load train data
header -  Size:  320.00KB , Shape:  (80, 4096)

Load val data
header -  Size:  40.00KB , Shape:  (10, 4096)

Load test data
header -  Size:  40.00KB , Shape:  (10, 4096)

Samples data:
X_train:  (80, 4096)
y_train:  (80,)
X_val:  (10, 4096)
y_val:  (10,)
X_test:  (10, 4096)
y_test:  (10,)

Time End: 2022-12-06 20:29:56.936385 - 0:00:04.633787
\Model fit...
Epoch 1/1000
3/3 [==============================] - 3s 782ms/step - loss: 2.1984 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.2000
Epoch 2/1000
3/3 [==============================] - 2s 617ms/step - loss: 2.1945 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1934 - val_sparse_categorical_accuracy: 0.2000
Epoch 3/1000
3/3 [==============================] - 2s 565ms/step - loss: 2.1918 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1914 - val_sparse_categorical_accuracy: 0.2000
Epoch 4/1000
3/3 [==============================] - 2s 573ms/step - loss: 2.1879 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1914 - val_sparse_categorical_accuracy: 0.2000
Epoch 5/1000
3/3 [==============================] - 2s 579ms/step - loss: 2.1844 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1895 - val_sparse_categorical_accuracy: 0.2000
Epoch 6/1000
3/3 [==============================] - 2s 599ms/step - loss: 2.1809 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.2000
Epoch 7/1000
3/3 [==============================] - 2s 601ms/step - loss: 2.1770 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.2000
Epoch 8/1000
3/3 [==============================] - 2s 574ms/step - loss: 2.1730 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1836 - val_sparse_categorical_accuracy: 0.2000
Epoch 9/1000
3/3 [==============================] - 2s 605ms/step - loss: 2.1703 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1836 - val_sparse_categorical_accuracy: 0.2000
Epoch 10/1000
3/3 [==============================] - 2s 566ms/step - loss: 2.1660 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1836 - val_sparse_categorical_accuracy: 0.2000
Epoch 11/1000
3/3 [==============================] - 2s 583ms/step - loss: 2.1629 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1836 - val_sparse_categorical_accuracy: 0.2000
Epoch 11: early stopping

Model evaluate...
1/1 [==============================] - 0s 26ms/step - loss: 2.2539 - sparse_categorical_accuracy: 0.1000

Model predict...
1/1 [==============================] - 0s 123ms/step

Store trains and tests results...

End round 1: 2022-12-06 20:30:19.841992 - 0:00:31.577668
--------------------------------------------------------------------------------

Start Round 2: 2022-12-06 20:30:19.842067

Load data to memory...

Time Start: 2022-12-06 20:30:19.916479

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Over sampling dataset...
X_over:  (61146, 1)
y_over:  (61146, 1)

Splitting Data into Training, Validation, and Testing...

mum_samples:  100
train: (80, 2)
 test: (10, 2)
  val: (10, 2)

Load train data
header -  Size:  320.00KB , Shape:  (80, 4096)

Load val data
header -  Size:  40.00KB , Shape:  (10, 4096)

Load test data
header -  Size:  40.00KB , Shape:  (10, 4096)

Samples data:
X_train:  (80, 4096)
y_train:  (80,)
X_val:  (10, 4096)
y_val:  (10,)
X_test:  (10, 4096)
y_test:  (10,)

Time End: 2022-12-06 20:30:20.536588 - 0:00:00.620067
\Model fit...
Epoch 1/1000
3/3 [==============================] - 3s 692ms/step - loss: 2.1969 - sparse_categorical_accuracy: 0.1000 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 2/1000
3/3 [==============================] - 2s 605ms/step - loss: 2.1953 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1934 - val_sparse_categorical_accuracy: 0.2000
Epoch 3/1000
3/3 [==============================] - 2s 579ms/step - loss: 2.1934 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1914 - val_sparse_categorical_accuracy: 0.2000
Epoch 4/1000
3/3 [==============================] - 2s 577ms/step - loss: 2.1910 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1895 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 5/1000
3/3 [==============================] - 2s 554ms/step - loss: 2.1891 - sparse_categorical_accuracy: 0.2250 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 6/1000
3/3 [==============================] - 2s 575ms/step - loss: 2.1859 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 7/1000
3/3 [==============================] - 2s 578ms/step - loss: 2.1855 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 8/1000
3/3 [==============================] - 2s 576ms/step - loss: 2.1824 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 9/1000
3/3 [==============================] - 2s 555ms/step - loss: 2.1809 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1855 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 10/1000
3/3 [==============================] - 2s 557ms/step - loss: 2.1785 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1875 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 11/1000
3/3 [==============================] - 2s 567ms/step - loss: 2.1770 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1855 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 12/1000
3/3 [==============================] - 2s 637ms/step - loss: 2.1750 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1836 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 12: early stopping

Model evaluate...
1/1 [==============================] - 0s 35ms/step - loss: 2.1836 - sparse_categorical_accuracy: 0.1000

Model predict...
1/1 [==============================] - 0s 138ms/step

Store trains and tests results...

End round 2: 2022-12-06 20:30:44.803190 - 0:00:24.961109
--------------------------------------------------------------------------------

Start Round 3: 2022-12-06 20:30:44.803269

Load data to memory...

Time Start: 2022-12-06 20:30:44.890833

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Over sampling dataset...
X_over:  (61146, 1)
y_over:  (61146, 1)

Splitting Data into Training, Validation, and Testing...

mum_samples:  100
train: (80, 2)
 test: (10, 2)
  val: (10, 2)

Load train data
header -  Size:  320.00KB , Shape:  (80, 4096)

Load val data
header -  Size:  40.00KB , Shape:  (10, 4096)

Load test data
header -  Size:  40.00KB , Shape:  (10, 4096)

Samples data:
X_train:  (80, 4096)
y_train:  (80,)
X_val:  (10, 4096)
y_val:  (10,)
X_test:  (10, 4096)
y_test:  (10,)

Time End: 2022-12-06 20:30:45.569804 - 0:00:00.678955
\Model fit...
Epoch 1/1000
3/3 [==============================] - 3s 681ms/step - loss: 2.1977 - sparse_categorical_accuracy: 0.0625 - val_loss: 2.2012 - val_sparse_categorical_accuracy: 0.1000
Epoch 2/1000
3/3 [==============================] - 2s 579ms/step - loss: 2.1965 - sparse_categorical_accuracy: 0.0750 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 3/1000
3/3 [==============================] - 2s 593ms/step - loss: 2.1953 - sparse_categorical_accuracy: 0.1375 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 4/1000
3/3 [==============================] - 3s 814ms/step - loss: 2.1941 - sparse_categorical_accuracy: 0.1375 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 5/1000
3/3 [==============================] - 2s 661ms/step - loss: 2.1918 - sparse_categorical_accuracy: 0.1375 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 6/1000
3/3 [==============================] - 2s 565ms/step - loss: 2.1914 - sparse_categorical_accuracy: 0.1375 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 7/1000
3/3 [==============================] - 2s 621ms/step - loss: 2.1895 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 8/1000
3/3 [==============================] - 3s 899ms/step - loss: 2.1883 - sparse_categorical_accuracy: 0.2250 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 9/1000
3/3 [==============================] - 2s 689ms/step - loss: 2.1855 - sparse_categorical_accuracy: 0.2500 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 10/1000
3/3 [==============================] - 2s 561ms/step - loss: 2.1840 - sparse_categorical_accuracy: 0.2750 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 11/1000
3/3 [==============================] - 2s 703ms/step - loss: 2.1840 - sparse_categorical_accuracy: 0.2500 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 11: early stopping

Model evaluate...
1/1 [==============================] - 0s 34ms/step - loss: 2.1816 - sparse_categorical_accuracy: 0.1000

Model predict...
1/1 [==============================] - 0s 122ms/step

Store trains and tests results...

End round 3: 2022-12-06 20:31:11.111717 - 0:00:26.308438
--------------------------------------------------------------------------------

Start Round 4: 2022-12-06 20:31:11.111784

Load data to memory...

Time Start: 2022-12-06 20:31:11.206678

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Over sampling dataset...
X_over:  (61146, 1)
y_over:  (61146, 1)

Splitting Data into Training, Validation, and Testing...

mum_samples:  100
train: (80, 2)
 test: (10, 2)
  val: (10, 2)

Load train data
header -  Size:  320.00KB , Shape:  (80, 4096)

Load val data
header -  Size:  40.00KB , Shape:  (10, 4096)

Load test data
header -  Size:  40.00KB , Shape:  (10, 4096)

Samples data:
X_train:  (80, 4096)
y_train:  (80,)
X_val:  (10, 4096)
y_val:  (10,)
X_test:  (10, 4096)
y_test:  (10,)

Time End: 2022-12-06 20:31:11.945289 - 0:00:00.738594
\Model fit...
Epoch 1/1000
3/3 [==============================] - 3s 652ms/step - loss: 2.1992 - sparse_categorical_accuracy: 0.0750 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.2000
Epoch 2/1000
3/3 [==============================] - 2s 569ms/step - loss: 2.1953 - sparse_categorical_accuracy: 0.1000 - val_loss: 2.2070 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 3/1000
3/3 [==============================] - 2s 774ms/step - loss: 2.1910 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.2148 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 4/1000
3/3 [==============================] - 3s 763ms/step - loss: 2.1867 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.2227 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 5/1000
3/3 [==============================] - 2s 737ms/step - loss: 2.1816 - sparse_categorical_accuracy: 0.1875 - val_loss: 2.2305 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 6/1000
3/3 [==============================] - 2s 573ms/step - loss: 2.1781 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2402 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 7/1000
3/3 [==============================] - 2s 746ms/step - loss: 2.1734 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2480 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 8/1000
3/3 [==============================] - 3s 835ms/step - loss: 2.1691 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2578 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 9/1000
3/3 [==============================] - 3s 774ms/step - loss: 2.1645 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2676 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 10/1000
3/3 [==============================] - 2s 723ms/step - loss: 2.1609 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2773 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 11/1000
3/3 [==============================] - 2s 582ms/step - loss: 2.1574 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2852 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 11: early stopping

Model evaluate...
1/1 [==============================] - 0s 27ms/step - loss: 2.2227 - sparse_categorical_accuracy: 0.1000

Model predict...
1/1 [==============================] - 0s 96ms/step

Store trains and tests results...

End round 4: 2022-12-06 20:31:38.601964 - 0:00:27.490171
--------------------------------------------------------------------------------

Start Round 5: 2022-12-06 20:31:38.602024

Load data to memory...

Time Start: 2022-12-06 20:31:38.679811

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Over sampling dataset...
X_over:  (61146, 1)
y_over:  (61146, 1)

Splitting Data into Training, Validation, and Testing...

mum_samples:  100
train: (80, 2)
 test: (10, 2)
  val: (10, 2)

Load train data
header -  Size:  320.00KB , Shape:  (80, 4096)

Load val data
header -  Size:  40.00KB , Shape:  (10, 4096)

Load test data
header -  Size:  40.00KB , Shape:  (10, 4096)

Samples data:
X_train:  (80, 4096)
y_train:  (80,)
X_val:  (10, 4096)
y_val:  (10,)
X_test:  (10, 4096)
y_test:  (10,)

Time End: 2022-12-06 20:31:39.358037 - 0:00:00.678205
\Model fit...
Epoch 1/1000
3/3 [==============================] - 4s 954ms/step - loss: 2.1980 - sparse_categorical_accuracy: 0.0750 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 2/1000
3/3 [==============================] - 3s 787ms/step - loss: 2.1957 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 3/1000
3/3 [==============================] - 3s 796ms/step - loss: 2.1945 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 4/1000
3/3 [==============================] - 3s 771ms/step - loss: 2.1937 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 5/1000
3/3 [==============================] - 2s 580ms/step - loss: 2.1918 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 6/1000
3/3 [==============================] - 2s 767ms/step - loss: 2.1895 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 7/1000
3/3 [==============================] - 3s 781ms/step - loss: 2.1879 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1973 - val_sparse_categorical_accuracy: 0.1000
Epoch 8/1000
3/3 [==============================] - 2s 766ms/step - loss: 2.1863 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 9/1000
3/3 [==============================] - 3s 775ms/step - loss: 2.1836 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 10/1000
3/3 [==============================] - 2s 728ms/step - loss: 2.1816 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.1992 - val_sparse_categorical_accuracy: 0.1000
Epoch 11/1000
3/3 [==============================] - 2s 579ms/step - loss: 2.1820 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2012 - val_sparse_categorical_accuracy: 0.1000
Epoch 12/1000
3/3 [==============================] - 2s 767ms/step - loss: 2.1797 - sparse_categorical_accuracy: 0.1625 - val_loss: 2.2031 - val_sparse_categorical_accuracy: 0.1000
Epoch 12: early stopping

Model evaluate...
1/1 [==============================] - 0s 33ms/step - loss: 2.2227 - sparse_categorical_accuracy: 0.1000

Model predict...
1/1 [==============================] - 0s 129ms/step

Store trains and tests results...

End round 5: 2022-12-06 20:32:09.572688 - 0:00:30.970650
--------------------------------------------------------------------------------

Start Round 6: 2022-12-06 20:32:09.572765

Load data to memory...

Time Start: 2022-12-06 20:32:09.675283

Loading original malware metadata from VirusShareSant...
X_exp:  (28617,)
y_exp:  (28617,)

Over sampling dataset...
X_over:  (61146, 1)
y_over:  (61146, 1)

Splitting Data into Training, Validation, and Testing...

mum_samples:  100
train: (80, 2)
 test: (10, 2)
  val: (10, 2)

Load train data
header -  Size:  320.00KB , Shape:  (80, 4096)

Load val data
header -  Size:  40.00KB , Shape:  (10, 4096)

Load test data
header -  Size:  40.00KB , Shape:  (10, 4096)

Samples data:
X_train:  (80, 4096)
y_train:  (80,)
X_val:  (10, 4096)
y_val:  (10,)
X_test:  (10, 4096)
y_test:  (10,)

Time End: 2022-12-06 20:32:10.424919 - 0:00:00.749621
\Model fit...
Epoch 1/1000
3/3 [==============================] - 4s 913ms/step - loss: 2.1969 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1934 - val_sparse_categorical_accuracy: 0.4000
Epoch 2/1000
3/3 [==============================] - 2s 579ms/step - loss: 2.1953 - sparse_categorical_accuracy: 0.1875 - val_loss: 2.1934 - val_sparse_categorical_accuracy: 0.4000
Epoch 3/1000
3/3 [==============================] - 2s 764ms/step - loss: 2.1918 - sparse_categorical_accuracy: 0.1750 - val_loss: 2.1914 - val_sparse_categorical_accuracy: 0.4000
Epoch 4/1000
3/3 [==============================] - 2s 755ms/step - loss: 2.1875 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1895 - val_sparse_categorical_accuracy: 0.4000
Epoch 5/1000
3/3 [==============================] - 2s 761ms/step - loss: 2.1840 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1836 - val_sparse_categorical_accuracy: 0.4000
Epoch 6/1000
3/3 [==============================] - 2s 757ms/step - loss: 2.1797 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1816 - val_sparse_categorical_accuracy: 0.4000
Epoch 7/1000
3/3 [==============================] - 2s 761ms/step - loss: 2.1750 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1777 - val_sparse_categorical_accuracy: 0.4000
Epoch 8/1000
3/3 [==============================] - 2s 718ms/step - loss: 2.1719 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1758 - val_sparse_categorical_accuracy: 0.4000
Epoch 9/1000
3/3 [==============================] - 2s 595ms/step - loss: 2.1684 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1738 - val_sparse_categorical_accuracy: 0.4000
Epoch 10/1000
3/3 [==============================] - 2s 762ms/step - loss: 2.1652 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1719 - val_sparse_categorical_accuracy: 0.4000
Epoch 11/1000
3/3 [==============================] - 2s 757ms/step - loss: 2.1633 - sparse_categorical_accuracy: 0.1500 - val_loss: 2.1699 - val_sparse_categorical_accuracy: 0.4000
Epoch 11: early stopping

Model evaluate...
1/1 [==============================] - 0s 33ms/step - loss: 2.2031 - sparse_categorical_accuracy: 0.1000

Model predict...
1/1 [==============================] - 0s 119ms/step

Store trains and tests results...

End round 6: 2022-12-06 20:32:38.095274 - 0:00:28.522499
--------------------------------------------------------------------------------

End training: 2022-12-06 20:32:38.095340 - 0:02:49.831092
```

Results:
```
(semantic-pe-cnn) $ python results.py --pe_part=header
--------------------------------------------------------------------------------
Semantic PE Malware Classifier (SPEMC) - Results
--------------------------------------------------------------------------------
Results for part:  header
0 train_data_men/header/2022-12-06-20-29-48
result_path in use:  train_data_men/header/2022-12-06-20-29-48
--------------------------------------------------------------------------------
       round  imput_len  params  epoca      loss        acc  val_loss    val_acc  eval_loss  eval_acc  pred %  duration       time
0          1       4096  215945     10  2.162891  17.500000  2.183594  20.000000   2.253906      10.0    10.0  00:00:31  31.544308
1          2       4096  215945     11  2.175000  15.000001  2.183594   0.000000   2.183594      10.0    10.0  00:00:24  24.929847
2          3       4096  215945     10  2.183984  25.000000  2.199219   0.000000   2.181641      10.0    10.0  00:00:26  26.277617
3          4       4096  215945     10  2.157422  16.249999  2.285156   0.000000   2.222656      10.0    10.0  00:00:27  27.461007
4          5       4096  215945     11  2.179688  16.249999  2.203125  10.000000   2.222656      10.0    10.0  00:00:30  30.937571
5          6       4096  215945     10  2.163281  15.000001  2.169922  40.000001   2.203125      10.0    10.0  00:00:28  28.491741
media      0       4096  215945     10  2.170378  17.500000  2.204102  11.666667   2.211263      10.0    10.0  00:00:28  28.273682

```
The results are storaged in the folder `train_data_men/header/2022-12-06-20-29-48`.
