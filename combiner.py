from glob import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser  # noqa
from keras.models import load_model  # noqa
import numpy as np  # noqa
from os import path # noqa


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-b',
                        '--basedir',
                        dest='base_dir',
                        help='Base directory of the model',
                        required=True)
    parser.add_argument('-i',
                        '--test-in',
                        dest='test_in_filename',
                        help='Test input file in .npy format',
                        required=True)
    parser.add_argument('-o',
                        '--test-out',
                        dest='test_out_filename',
                        help='Test output file in .npy format',
                        required=True)
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
                        '--clusters-size',
                        dest='clusters_sz',
                        default='10',
                        help='Number of neurons per cluster (default: 10).',
                        required=False)

    args = parser.parse_args()

    basename = path.basename(args.base_dir)
    module_files = glob(path.join(args.base_dir,
                                  'modules-optimized-%d-%.2f-%.2f' % (int(args.clusters_sz),
                                                                      float(args.threshold),
                                                                      float(args.weight_fraction)),
                                  '*.keras'))
    modules = dict()
    for module_file in module_files:
        module_index = int(path.basename(module_file).split('-')[1].split('.')[0])
        print('Loading module model for class %d...' % module_index)
        modules[module_index] = load_model(module_file)
    print('Loading original model...')
    original_model = load_model(path.join(args.base_dir, '%s.h5' % basename))
    print('Loading datasets 1/2...')
    input_data = np.load(args.test_in_filename)
    print('Loading datasets 2/2...')
    output_data = np.load(args.test_out_filename)
    print('Calculating train accuracy of the original model...')
    original_acc = original_model.evaluate(input_data, output_data, verbose=0)[1]
    predictions = np.zeros(output_data.shape)
    for module_index, module_model in modules.items():
        print('Applying module model %d on the training dataset...' % module_index)
        predictions[:, module_index] = module_model.predict(input_data, verbose=0)[:, 1]
    true_labels = np.argmax(output_data, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == true_labels)
    combined_accuracy = correct_predictions / len(true_labels)
    print('---------------')
    print('Combined Acc: %f' % combined_accuracy)
    print('Original Model Acc: %f' % original_acc)
