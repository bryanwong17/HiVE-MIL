import os
import argparse
import pandas as pd
import h5py
import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict

def filter_high_mag_coords(high_mag_coords, low_mag_coords, low_mag_original_size=2048):
    """
    Filter high-magnification coordinates to map them to low-magnification regions.
    """
    filtered_high_mag_coords_dict = {}
    for x_low, y_low in low_mag_coords:
        x_min, x_max = x_low, x_low + low_mag_original_size
        y_min, y_max = y_low, y_low + low_mag_original_size

        filtered_coords = [
            (x_high, y_high)
            for x_high, y_high in high_mag_coords
            if x_min <= x_high < x_max and y_min <= y_high < y_max
        ]
        if filtered_coords:
            filtered_high_mag_coords_dict[(x_low, y_low)] = filtered_coords
    return filtered_high_mag_coords_dict

def filter_high_mag_coords(high_mag_coords, low_mag_coords, low_mag_original_size=2048):
    high_mag_coords = np.array(high_mag_coords)
    low_mag_coords = np.array(low_mag_coords)

    filtered_high_mag_coords_dict = {}

    for x_low, y_low in low_mag_coords:
        x_min, x_max = x_low, x_low + low_mag_original_size
        y_min, y_max = y_low, y_low + low_mag_original_size

        mask = (
            (high_mag_coords[:, 0] >= x_min) & (high_mag_coords[:, 0] < x_max) &
            (high_mag_coords[:, 1] >= y_min) & (high_mag_coords[:, 1] < y_max)
        )
        filtered_coords = high_mag_coords[mask].tolist()
        
        if filtered_coords:
            filtered_high_mag_coords_dict[(x_low, y_low)] = filtered_coords
    
    return filtered_high_mag_coords_dict
    
def get_features_and_coords_from_h5(h5_file):
    """
    Extract features and coordinates from an .h5 file.
    """
    with h5py.File(h5_file, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:]
    return features, coords

def group_features(low_mag_coords, high_mag_features, high_mag_coords, max_patches, low_mag_original_size):
    """
    Group high-mag features under low-mag regions.
    """

    filtered_high_mag_coords = filter_high_mag_coords(high_mag_coords, low_mag_coords, low_mag_original_size)

    embeddings_by_low_mag = defaultdict(list)
    coords_by_low_mag = defaultdict(list)

    high_mag_coord_to_feature = {tuple(coord): feature for coord, feature in zip(high_mag_coords, high_mag_features)}

    for low_mag_coord, filtered_coords in filtered_high_mag_coords.items():
        for coord in filtered_coords:
            coord = tuple(np.int64(c) for c in coord)
            if coord in high_mag_coord_to_feature:
                embeddings_by_low_mag[low_mag_coord].append(high_mag_coord_to_feature[coord])
                coords_by_low_mag[low_mag_coord].append(coord)

    slide_embeddings, slide_coords = [], []

    for low_mag_coord, region_embeddings in embeddings_by_low_mag.items():
        region_embeddings = np.vstack(region_embeddings)
        region_coords = np.vstack(coords_by_low_mag[low_mag_coord])

        num_patches = region_embeddings.shape[0]
        if num_patches > max_patches:
            selected_indices = np.random.choice(num_patches, max_patches, replace=False)
            region_embeddings = region_embeddings[selected_indices]
            region_coords = region_coords[selected_indices]
        elif num_patches < max_patches:
            pad_size = max_patches - num_patches
            region_embeddings = np.pad(region_embeddings, ((0, pad_size), (0, 0)), mode="constant")
            region_coords = np.pad(region_coords, ((0, pad_size), (0, 0)), mode="constant")

        slide_embeddings.append(region_embeddings)
        slide_coords.append(region_coords)

    slide_embeddings = np.stack(slide_embeddings)  # [R, max_patches, D]
    slide_coords = np.stack(slide_coords)  # [R, max_patches, 2]

    return slide_embeddings, slide_coords

def main():
    parser = argparse.ArgumentParser(description='Group high-mag features under low-mag regions based on coordinates.')
    parser.add_argument('--dataset-root-path', type=str, default='/path/to/dataset', help='Root path to the dataset')
    parser.add_argument('--dataset-name', type=str, choices=['tcga_nsclc', 'tcga_brca', 'tcga_rcc'], default='tcga_nsclc')
    parser.add_argument('--feature-extractor-name', type=str, default='quiltnet')
    parser.add_argument('--low-mag', type=str, default='5x')
    parser.add_argument('--high-mag', type=str, default='20x')
    parser.add_argument('--max-patches', type=int, default=16)
    args = parser.parse_args()

    start_time = time.time()

    df = pd.read_csv(os.path.join('dataset_csv', f'{args.dataset_name}.csv'))
    slide_id_to_level0_mag = {row["slide_id"]: row["level0_mag"] for _, row in df.iterrows()}

    low_mag_dir = os.path.join(args.dataset_root_path, f'{args.dataset_name}_{args.feature_extractor_name}', f'{args.feature_extractor_name}_{args.low_mag}')
    high_mag_dir = os.path.join(args.dataset_root_path, f'{args.dataset_name}_{args.feature_extractor_name}', f'{args.feature_extractor_name}_{args.high_mag}')
    output_dir = os.path.join(args.dataset_root_path, f'{args.dataset_name}_{args.feature_extractor_name}', f'hierarchical_{args.feature_extractor_name}_{args.low_mag}_{args.high_mag}')
    os.makedirs(output_dir, exist_ok=True)

    for slide_id in tqdm(os.listdir(low_mag_dir), desc=f"Processing slides"):
        low_mag_file = os.path.join(low_mag_dir, slide_id) # slide id with .h5 extension
        high_mag_file = os.path.join(high_mag_dir, slide_id)
        output_file = os.path.join(output_dir, slide_id)

        if os.path.exists(output_file):
            print(f"Output file already exists for slide {slide_id}, skipping...")
            continue

        if not os.path.exists(high_mag_file):
            print(f"Missing high-mag file for slide {slide_id}, skipping...")
            continue

        slide_id_base = os.path.splitext(slide_id)[0] # slide id without .h5 extension
        level0_mag = slide_id_to_level0_mag.get(slide_id_base, None)
        if level0_mag is None:
            print(f"Slide {slide_id} not found in CSV, skipping...")
            continue

        low_mag_original_size = 2048 if level0_mag == 40 else 1024 if level0_mag == 20 else None
        if low_mag_original_size is None:
            print(f"Invalid level0_mag {level0_mag} for slide {slide_id}, skipping...")
            continue

        low_mag_features, low_mag_coords = get_features_and_coords_from_h5(low_mag_file)
        high_mag_features, high_mag_coords = get_features_and_coords_from_h5(high_mag_file)


        slide_embeddings, slide_coords = group_features(
            low_mag_coords, high_mag_features, high_mag_coords, args.max_patches, low_mag_original_size
        )

        with h5py.File(output_file, 'w') as f:
            f['features'] = slide_embeddings
            f['coords'] = slide_coords

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Total time taken: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

if __name__ == '__main__':
    main()