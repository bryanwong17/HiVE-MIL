import os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from patch_extraction_utils import create_embeddings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Configurations for feature extraction')
parser.add_argument('--dataset', type=str, default='TCGA')
parser.add_argument('--patches_path', type=str)
parser.add_argument('--library_path', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()

dataset = args.dataset
patches_path = args.patches_path
library_path = args.library_path
model_name = args.model_name

os.makedirs(library_path, exist_ok=True)

create_embeddings(patch_datasets=patches_path, dataset=dataset, embeddings_dir=library_path,
                  enc_name=model_name, batch_size=args.batch_size, gpu_index=args.gpu_index)