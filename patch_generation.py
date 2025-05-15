import os
import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

dataset_name = 'tcga_nsclc'
root_folder = f'/path/to/dataset/{dataset_name}'
slide_folder = f'{root_folder}/WSI'
define_patch_size = 512 # 512: 20x, 2048: 5x
patch_folder = os.path.join(root_folder, 'patches_'+str(define_patch_size))

if(define_patch_size == 2048):
    scale = 8
    save_name = '5x'
elif(define_patch_size == 1024):
    scale = 4
    save_name = '10x'
elif(define_patch_size == 512):
    scale = 2
    save_name = '20x'

save_folder = os.path.join(root_folder, f'patches_{save_name}')

if(not os.path.exists(root_folder)):
    os.makedirs(root_folder)

def generate_patch(patch_file_name):
    patch_path = os.path.join(patch_folder, patch_file_name)
    slide_path = os.path.join(slide_folder, patch_file_name.replace('h5', 'svs'))

    f = h5py.File(patch_path, 'r')
    coords = f['coords']
    patch_level = coords.attrs['patch_level']
    patch_size = coords.attrs['patch_size']
    coords_list = coords[:].tolist()
   
    slide = openslide.open_slide(slide_path)

    try:
        magnification = int(float(slide.properties['aperio.AppMag'])) # for TCGA datasets to get magnification directly
    except:
        # for camelyon16 to get magnification from mpp
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0))
		
        # Average microns per pixel for x and y
        if mpp_x > 0 and mpp_y > 0:
            actual_mpp = (mpp_x + mpp_y) / 2

            # Reference mpp values for different magnifications
            mpp_40x = 0.25
            mpp_20x = 0.5

            # Check which magnification is closer to the actual mpp
            if (mpp_40x - actual_mpp) < (mpp_20x - actual_mpp):
                magnification = 40
            else:
                magnification = 20
        else:
            magnification = 40
    save_path = os.path.join(save_folder, patch_file_name.replace('.h5', ''))
    
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
        if(magnification == 40):
            resized_patch_size = int(patch_size/scale)
        elif(magnification == 20):
            resized_patch_size = int(patch_size/(scale/2))
    
        for x, y in tqdm(coords_list):
            x, y = int(x), int(y)
            patch = slide.read_region((x,y), int(patch_level), (int(patch_size), int(patch_size))).convert('RGB')
            patch = patch.resize((resized_patch_size, resized_patch_size))
            patch_name = str(x) + '_' + str(y) + '.png'
            patch_save_path = os.path.join(save_path, patch_name)
            patch.save(patch_save_path)
    else:
        print(patch_file_name + ': has been processed !')

pool = ThreadPoolExecutor(max_workers=16)
all_file_names = np.array(os.listdir(patch_folder))
for patch_file_name in all_file_names:
    pool.submit(generate_patch, patch_file_name)
pool.shutdown(wait=True)
