from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_model.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np

import os

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default='/path/to/dataset', help='data directory')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--feature_extractor', type=str, choices=['plip', 'quiltnet', 'conch'], default='plip', help='feature extractor model')
parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
parser.add_argument('--few_shot_num', type=int, default=16, help='number of few-shot samples')
parser.add_argument('--high_mag', type=str, default='20x', help='high magnification')
parser.add_argument('--low_mag', type=str, default='5x', help='low magnification')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, default='HiVE_MIL', help='The name of MIL model or aggregator')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--task', type=str)
parser.add_argument('--text_prompt', type=str, default=None)
parser.add_argument('--prototype_number', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda:0')

# HiVE_MIL specific arguments
parser.add_argument('--LLM', type=str, default='gpt4o', help='The name of LLM model')
parser.add_argument('--num-context-tokens', type=int, default=16, help='The number of context tokens')
parser.add_argument('--class-specific-token', action='store_true', help='Whether to use class-specific token')
parser.add_argument('--class-token-position', type=str, default='end', choices=['front', 'middle', 'end'], help='The position of class token')
parser.add_argument('--num-low-mag-texts', type=int, default=4, help='The number of texts at low magnification level (parent)')
parser.add_argument('--num-high-mag-subtexts', type=int, default=3, help='The number of subtexts for each text at high magnification level (children)')
parser.add_argument('--filter-alpha', type=float, default=0.5, help='Filter alpha for filtering patches based on text similarity')
parser.add_argument('--contrastive_lambda', type=float, default=0.5, help='Contrastive loss weight (default: 0.1)')

args = parser.parse_args()

import warnings
warnings.simplefilter("always") 
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
seed_torch(args.seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


args.exp_code = '{}_{}_{}_{}'.format(args.task, args.feature_extractor, args.model_type, args.few_shot_num)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'feature_extractor': args.feature_extractor,
            'feature_dim': args.feature_dim,
            'model_type': args.model_type,
            'few_shot_num': args.few_shot_num,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'mode': args.mode,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

print('\nLoad Dataset')

if args.task == 'tcga_nsclc':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_nsclc.csv',
                                  model_type = args.model_type,
                                  feature_extractor = args.feature_extractor,
                                  data_root = f"{args.data_root_dir}/{args.task}_{args.feature_extractor}",
                                  data_dir_s = os.path.join(args.data_root_dir, f"{args.task}_{args.feature_extractor}", str(args.feature_extractor + '_{}'.format(args.low_mag))),
                                  data_dir_l = os.path.join(args.data_root_dir, f"{args.task}_{args.feature_extractor}", str(args.feature_extractor + '_{}'.format(args.high_mag))),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'tcga_brca':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_brca.csv',
                                  model_type = args.model_type,
                                  feature_extractor = args.feature_extractor,
                                  data_root = f"{args.data_root_dir}/{args.task}_{args.feature_extractor}",
                                  data_dir_s = os.path.join(args.data_root_dir, f"{args.task}_{args.feature_extractor}", str(args.feature_extractor + '_{}'.format(args.low_mag))),
                                  data_dir_l = os.path.join(args.data_root_dir, f"{args.task}_{args.feature_extractor}", str(args.feature_extractor + '_{}'.format(args.high_mag))),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'IDC':0, 'ILC':1},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'tcga_rcc':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_rcc.csv',
                                  model_type = args.model_type,
                                  feature_extractor = args.feature_extractor,
                                  data_root = f"{args.data_root_dir}/{args.task}_{args.feature_extractor}",
                                  data_dir_s = os.path.join(args.data_root_dir, f"{args.task}_{args.feature_extractor}", str(args.feature_extractor + '_{}'.format(args.low_mag))),
                                  data_dir_l = os.path.join(args.data_root_dir, f"{args.task}_{args.feature_extractor}", str(args.feature_extractor + '_{}'.format(args.high_mag))),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'CCRCC':0, 'CHRCC':1, 'PRCC':2},
                                  patient_strat= False,
                                  ignore=[])          
else:
    raise NotImplementedError

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, args.task, args.feature_extractor, str(args.few_shot_num), str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.split_dir = os.path.join('splits', args.task, args.task+'_{}'.format(int(args.few_shot_num)))

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)

        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
                              'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
                              'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


