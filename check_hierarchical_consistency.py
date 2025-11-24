import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fix mismatched 5x coordinates not covered by 20x patches.")
    parser.add_argument('--dataset-root-path', type=str, default='/path/to/dataset', help='Root path to the dataset')
    parser.add_argument('--dataset-name', type=str, choices=['tcga_nsclc', 'tcga_brca', 'tcga_rcc'], default='tcga_nsclc')
    parser.add_argument('--feature-extractor-name', type=str, default='quiltnet')
    parser.add_argument('--csv-path', type=str, default='dataset_csv', help='Directory containing dataset CSV file')
    return parser.parse_args()

def get_features_and_coords_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        return f['features'][:], f['coords'][:]

def find_missing_coords(high_mag_coords, low_mag_coords, low_mag_original_size=2048):
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

    low_mag_set = set(tuple(coord) for coord in low_mag_coords)
    filtered_set = set(filtered_high_mag_coords_dict.keys())
    missing_coords = low_mag_set - filtered_set
    return missing_coords

def process_slide(slide_file):
    slide_id = slide_file.split('.h5')[0]
    path_low_mag = os.path.join(low_mag_dir, slide_file)
    path_high_mag = os.path.join(high_mag_dir, slide_file)
    path_hierarchical_mag = os.path.join(hierarchical_mag_dir, slide_file)

    level0_mag = slide_id_to_level0_mag.get(slide_id, None)
    if level0_mag is None:
        return f"Slide {slide_id} not found in CSV, skipping..."

    low_mag_original_size = 2048 if level0_mag == 40 else 1024 if level0_mag == 20 else None
    if low_mag_original_size is None:
        return f"Invalid level0_mag {level0_mag} for slide {slide_id}, skipping..."

    if os.path.exists(path_high_mag):
        _, coords_high_mag = get_features_and_coords_from_h5(path_high_mag)
        features_low_mag, coords_low_mag = get_features_and_coords_from_h5(path_low_mag)
        features_hierarchical_mag, _ = get_features_and_coords_from_h5(path_hierarchical_mag)

        if features_low_mag.shape[0] != features_hierarchical_mag.shape[0]:
            missing_coords = find_missing_coords(coords_high_mag, coords_low_mag, low_mag_original_size)

            if missing_coords:
                coords_to_remove = list(missing_coords)
                mask = ~np.any(np.all(coords_low_mag[:, None] == coords_to_remove, axis=2), axis=1)
                coords_filtered = coords_low_mag[mask]
                features_filtered = features_low_mag[mask]

                with h5py.File(path_low_mag, 'w') as hf:
                    hf.create_dataset('coords', data=coords_filtered)
                    hf.create_dataset('features', data=features_filtered)

                return f"Fixed mismatches for slide {slide_id}"
    return None

if __name__ == "__main__":
    args = parse_args()

    dataset_root_path = args.dataset_root_path
    dataset_name = args.dataset_name
    feature_extractor_name = args.feature_extractor_name

    low_mag_dir = os.path.join(dataset_root_path, f'{args.dataset_name}_{args.feature_extractor_name}', f'{feature_extractor_name}_5x')
    high_mag_dir = os.path.join(dataset_root_path, f'{args.dataset_name}_{args.feature_extractor_name}', f'{feature_extractor_name}_20x')
    hierarchical_mag_dir = os.path.join(dataset_root_path, f'{args.dataset_name}_{args.feature_extractor_name}', f'hierarchical_{feature_extractor_name}_5x_20x')

    df = pd.read_csv(os.path.join(args.csv_path, f'{dataset_name}.csv'))
    slide_id_to_level0_mag = {row["slide_id"]: row["level0_mag"] for _, row in df.iterrows()}

    low_mag_files = [f for f in os.listdir(low_mag_dir) if f.endswith('.h5')]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_slide, low_mag_files), total=len(low_mag_files), desc="Processing slides"))

    for res in results:
        if res:
            print(res)
