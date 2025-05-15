### Dependencies
# Base Dependencies
import os
# LinAlg / Stats / Plotting Dependencies
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.nn as nn
import torch.multiprocessing
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
torch.multiprocessing.set_sharing_strategy('file_system')

from conch.open_clip_custom import create_model_from_pretrained

def eval_transforms_clip(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Resize((224, 224)),
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val

def eval_transforms_histopathology():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Resize((224, 224)),
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val

def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val


# def torchvision_ssl_encoder(name: str, pretrained: bool = False, return_all_feature_maps: bool = False):
#     pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
#     pretrained_model.fc = nn.Identity()
#     return pretrained_model


def save_embeddings(model, fname, dataloader, enc_name, overwrite=False, device='cuda:0'):

    if os.path.isfile('%s.h5' % fname) and (overwrite == False):
        return None

    embeddings, coords, file_names = [], [], []

    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            if enc_name == 'clip_ViTB16' or enc_name == 'clip_ViTB32' or enc_name == 'plip' or enc_name == 'quiltnet':
                embeddings.append(model.get_image_features(batch).detach().cpu().numpy().squeeze())
            elif enc_name == 'conch':
                embeds = model.encode_image(batch, proj_contrast=True, normalize=True).detach().cpu().numpy().squeeze()
                embeddings.append(embeds)
            else:
                embeddings.append(model(batch).detach().cpu().numpy().squeeze())
            file_names.append(coord)

    for file_name in file_names:
        for coord in file_name:
            coord = coord.rstrip('.png').split('_')
            coords.append([int(coord[0]), int(coord[1])])

    # print(fname)

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    f = h5py.File(fname+'.h5', 'w')
    f['features'] = embeddings
    f['coords'] = coords
    f.close()


def create_embeddings(embeddings_dir, enc_name, dataset, batch_size, gpu_index, save_patches=False,
                      patch_datasets='path/to/patch/datasets', assets_dir ='./ckpts/',
                      disentangle=-1, stage=-1):
    print("Extracting Features for '%s' via '%s'" % (dataset, enc_name))

    if enc_name == 'plip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('vinid/plip')
        eval_t = eval_transforms_histopathology()
    
    elif enc_name == 'quiltnet':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('wisdomik/QuiltNet-B-32')
        eval_t = eval_transforms_histopathology()

    elif enc_name == 'conch':
        model_name = 'conch_ViT-B-16'
        model_path = "hf_hub:MahmoodLab/conch"
        hf_auth_token = "" # Add your Hugging Face auth token here
        model, eval_t = create_model_from_pretrained(model_name, model_path, hf_auth_token=hf_auth_token)

    else:
        pass
   
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    non_parallel_encoders = {'clip_RN50', 'clip_ViTB16', 'clip_ViTB32', 'plip', 'conch', 'quiltnet'}
    if enc_name not in non_parallel_encoders:
        model = torch.nn.DataParallel(model)

    model.eval()

    if 'dino' in enc_name:
        _model = model
        if stage == -1:
            model = _model
        else:
            model = lambda x: torch.cat([x[:, 0] for x in _model.get_intermediate_layers(x, stage)], dim=-1)

    if stage != -1:
        _stage = '_s%d' % stage
    else:
        _stage = ''

    # pool = ThreadPoolExecutor(max_workers=48)
    for wsi_name in tqdm(os.listdir(patch_datasets)):
        dataset = PatchesDataset(os.path.join(patch_datasets, wsi_name), transform=eval_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        fname = os.path.join(embeddings_dir, wsi_name)

        # Check if the embedding file already exists
        if os.path.isfile(f'{fname}.h5'):
            print(f"Skipping {fname}.h5 as it already exists.")
            continue  # Skip processing if the file exists
        
        if(not os.path.exists(fname)):
            save_embeddings(model=model,
                            fname=fname,
                            dataloader=dataloader,
                            enc_name=enc_name,
                            device=device)
            # args = [model, fname, dataloader]
            # pool.submit(lambda p: save_embeddings(*p), args)
    # pool.shutdown(wait=True)


class PatchesDataset(Dataset):
    def __init__(self, file_path, transform=None):
        file_names = os.listdir(file_path)
        imgs = []
        coords = []
        for file_name in file_names:
            imgs.append(os.path.join(file_path, file_name))
            coords.append(file_name)
        self.imgs = imgs
        self.coords = coords
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)
        
