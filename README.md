# Incite: Decomposition of Deep Neural Networks into Modules via Mutation Analysis

This repository contains the source code of Incite implementation.
Incite is a DNN decomposition technique that uses neuron mutation to quantify the contribution of the neurons to a
given output of a model.
Then, for each output of the model, a subgraph induced by the nodes with highest contribution scores for that output
are selected and extracted as a module.
This approach is agnostic of the type of the model and the activation functions used in it, and is applicable to not
just classifiers, but to regression models as well.

## System Requirements and Basic Setup
For using Incite, your machine needs to have the following requirements met.
Other system configurations may or may not work.

* Ubuntu Linux 22.04.4 LTS 64-bit, or macOS 14.5 on Apple M2 Pro
* Miniconda 23.11.0
* Git version control system

#### Step 1: Checkout the repository
Run the following commands in a terminal window.
```shell
$ git clone https://github.com/ali-ghanbari/incite-issta24
$ cd incite-issta24
```

#### Step 2: Setting up the Python virtual environment
If you are using Incite for the first time, run the following command in a terminal window, and answer the question(s)
issued by Miniconda, to get your conda virtual environment created.
If you have already created incite environment, you may skip this command.
```shell
$ conda create -n incite python=3.9
```
Once the virtual environment has been created, run the following commands to activate the environment and install
the required dependencies.
```shell
$ conda activate incite
$ pip install -r requirements.txt
```

## Incite Interface
Please follow the instructions given in the previous section to install Python 3.9 and all the dependencies required by
Incite in a conda virtual environment.
The file `main.py` is the entry point for Incite.
Basic usage of Incite is as follows, where optional parameter are placed inside square brackets.

```text
main.py [-h] -m MODEL_FILENAME [-tt TASK_TYPE] -tri TRAIN_IN_FILENAME -tro TRAIN_OUT_FILENAME -tsi TEST_IN_FILENAME
    -tso TEST_OUT_FILENAME [-vs VALIDATION_SPLIT] [-th THRESHOLD] [-w WEIGHT_FRACTION] [-c CLUSTER_SIZE]
    [-o OPTIMIZER] [-e EPOCHS] [-p PATIENCE] [-dev DEVICE]
```

Detailed information of the parameters are as follows.
```text
  -h, --help            show this help message and exit
  -m MODEL_FILENAME, --model MODEL_FILENAME
                        Model file in .h5/.keras format. This is the mode to be decomposed.
  -tt TASK_TYPE, --task-type TASK_TYPE
                        Task type for the model. Possible values: 'class' for classification, 'reg' for regression (default: class)
  -tri TRAIN_IN_FILENAME, --train-in TRAIN_IN_FILENAME
                        Train input file in .npy format. This dataset will be used for weight adjustment.
  -tro TRAIN_OUT_FILENAME, --train-out TRAIN_OUT_FILENAME
                        Train output file in .npy format. This dataset will be used for weight adjustment.
  -tsi TEST_IN_FILENAME, --test-in TEST_IN_FILENAME
                        Test input file in .npy format. This dataset will NOT be used for weight adjustment.
  -tso TEST_OUT_FILENAME, --test-out TEST_OUT_FILENAME
                        Test output file in .npy format. This dataset will NOT be used for weight adjustment.
  -vs VALIDATION_SPLIT, --validation-split VALIDATION_SPLIT
                        Validation split. The percentage of train dataset to be used as test datat during weight adjustment (default: 0.1)
  -th THRESHOLD, --threshold THRESHOLD
                        Percentage of top most impactful neuron clusters to be selected from each layer (default: 0.6)
  -w WEIGHT_FRACTION, --weight-fraction WEIGHT_FRACTION
                        The fraction to add/subtract when mutating the weights (default: 0.1)
  -c CLUSTER_SIZE, --cluster-size CLUSTER_SIZE
                        Number neurons per neuron cluster.
  -o OPTIMIZER, --optimizer OPTIMIZER
                        Optimizer for re-training the last layer of the slice, e.g., 'adam', 'sgd', etc.(default: adam)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs used for re-training the last layer of the slice (default: 64)
  -p PATIENCE, --patience PATIENCE
                        Patience level for early stopping during model weight adjustment (default: -1, i.e., infinite patience)
  -dev DEVICE, --device DEVICE
                        0, 1, ... for GPU, -1 for CPU, and -2 for automatic device selection (default: -2)
```

You can always pass `-h` (or `--help`) flag to the tool to see a short description of the flag and their default values.

Once you decompose a DNN model into modules the modules, you can find the module models saved as `.keras` files under
a folder named `modules-optimized-*`.
You may use the `combiner.py` (in case of classifiers) or `combiner_reg.py` (in case of regression models) to combine
the modules.

## Research
Incite is an implementation of a technique presented in ISSTA 2024.
If you use Incite in your research, please use the following BibTeX entry.

```text
@inproceedings{ghanbari2024incite,
  title={Decomposition of Deep Neural Networks into Modules via Mutation Analysis},
  author={Ghanbari, Ali},
  booktitle={Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis},
  note={To appear},
  year={2024}
}
```
