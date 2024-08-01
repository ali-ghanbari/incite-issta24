import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser # noqa
from entry import Entry # noqa


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        dest='model_filename',
                        help='Model file in .h5/.keras format. This is the mode to be decomposed.',
                        required=True)
    parser.add_argument('-tt',
                        '--task-type',
                        dest='task_type',
                        help='Task type for the model. Possible values: \'class\' for classification, '
                             '\'reg\' for regression (default: class)',
                        default='class',
                        required=False)
    parser.add_argument('-tri',
                        '--train-in',
                        dest='train_in_filename',
                        help='Train input file in .npy format. This dataset will be used for weight adjustment.',
                        required=True)
    parser.add_argument('-tro',
                        '--train-out',
                        dest='train_out_filename',
                        help='Train output file in .npy format. This dataset will be used for weight adjustment.',
                        required=True)
    parser.add_argument('-tsi',
                        '--test-in',
                        dest='test_in_filename',
                        help='Test input file in .npy format. This dataset will NOT be used for weight adjustment.',
                        required=True)
    parser.add_argument('-tso',
                        '--test-out',
                        dest='test_out_filename',
                        help='Test output file in .npy format. This dataset will NOT be used for weight adjustment.',
                        required=True)
    parser.add_argument('-vs',
                        '--validation-split',
                        dest='validation_split',
                        default='0.1',
                        help='Validation split. The percentage of train dataset to be used as test datat '
                             'during weight adjustment (default: 0.1)',
                        required=False)
    parser.add_argument('-th',
                        '--threshold',
                        dest='threshold',
                        default='0.6',
                        help='Percentage of top most impactful neuron clusters to be selected from each layer '
                             '(default: 0.6)',
                        required=False)
    parser.add_argument('-w',
                        '--weight-fraction',
                        dest='weight_fraction',
                        default='0.1',
                        help='The fraction to add/subtract when mutating the weights (default: 0.1)',
                        required=False)
    parser.add_argument('-c',
                        '--cluster-size',
                        dest='cluster_size',
                        default='10',
                        help='Number neurons per neuron cluster (default: 10)',
                        required=False)
    parser.add_argument('-o',
                        '--optimizer',
                        dest='optimizer',
                        help='Optimizer for re-training the last layer of the slice, e.g., \'adam\', \'sgd\', etc. '
                             '(default: adam)',
                        default='adam',
                        required=False)
    parser.add_argument('-e',
                        '--epochs',
                        dest='epochs',
                        help='Number of epochs used for re-training the last layer of the slice (default: 128)',
                        default='128',
                        required=False)
    parser.add_argument('-p',
                        '--patience',
                        dest='patience',
                        help='Patience level for early stopping during model weight adjustment (default: -1, i.e., '
                             'infinite patience)',
                        default='-1',
                        required=False)
    parser.add_argument('-dev',
                        '--device',
                        dest='device',
                        help='0, 1, ... for GPU, -1 for CPU, and -2 for automatic device selection (default: -2)',
                        default='-2',
                        required=False)
    args = parser.parse_args()

    if int(args.device) > -2:
        print('Device %s' % args.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    entry = Entry()
    print('Loading model...')
    entry.load_model(args.model_filename)
    print('Loading datasets 1/4...')
    entry.load_train_inputs(args.train_in_filename)
    print('Loading datasets 2/4...')
    entry.load_train_outputs(args.train_out_filename)
    print('Loading datasets 3/4...')
    entry.load_test_inputs(args.test_in_filename)
    print('Loading datasets 4/4...')
    entry.load_test_outputs(args.test_out_filename)
    print('Initializing...')
    task_type = args.task_type
    if task_type != 'class' and task_type != 'reg':
        raise ValueError('Invalid task type %s. Possible values: \'class\' or \'reg\'' % task_type)
    entry.set_task_type(task_type)
    entry.set_validation_split(float(args.validation_split))
    entry.set_mutation_weight_fraction(float(args.weight_fraction))
    entry.set_cluster_size(int(args.cluster_size))
    entry.set_cluster_selection_threshold(float(args.threshold))
    entry.set_optimizer(args.optimizer)
    entry.set_epochs(int(args.epochs))
    patience = int(args.patience)
    if patience == 0:
        print('Warning: ignored patience value of 0.')
    entry.set_patience(patience)
    entry.run()
