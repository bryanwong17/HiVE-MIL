# HiVE-MIL

**Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling**  
*Under Review*

![Framework](framework.pdf)

Vision-language models (VLMs) have recently been integrated into multiple instance learning (MIL) frameworks to address the challenge of few-shot, weakly supervised classification of whole slide images (WSIs). A key trend involves leveraging multi-scale information to better represent hierarchical tissue structures. However, existing methods often face two key limitations: (1) insufficient modeling of interactions within the same modalities across scales (e.g., 5x and 20x) and (2) inadequate alignment between visual and textual modalities on the same scale. To address these gaps, we propose HiVE-MIL, a hierarchical vision-language framework that constructs a unified graph consisting of (1) parent–child links between coarse (5x) and fine (20x) visual/textual nodes to capture hierarchical relationships, and (2) heterogeneous intra-scale edges linking visual and textual nodes on the same scale. To further enhance semantic consistency, HiVE-MIL incorporates a two-stage, text-guided dynamic filtering mechanism that removes weakly correlated patch–text pairs, and introduces a hierarchical contrastive loss to align textual semantics across scales. Extensive experiments on TCGA breast, lung, and kidney cancer datasets demonstrate that HiVE-MIL consistently outperforms both traditional MIL and recent VLM-based MIL approaches, achieving gains of up to 4.1\% in macro F1 under 16-shot settings. Our results demonstrate the value of jointly modeling hierarchical structure and multimodal alignment for efficient and scalable learning from limited pathology data.

---

## 1. Pre-requisites

Install the required dependencies:
```
conda create -n hivemil python=3.9.21
conda activate hivemil
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


## 2. Datasets

HiVE-MIL is evaluated on three publicly available WSI datasets from The Cancer Genome Atlas (TCGA):

- **NSCLC (Lung Cancer)**: LUAD vs. LUSC  
- **BRCA (Breast Cancer)**: IDC vs. ILC  
- **RCC (Kidney Cancer)**: CCRCC vs. PRCC vs. CHRCC

## 3. Construct Hierarchical Patch Features

To link corresponding 5× and 20× patches via absolute coordinates, run:


```
python create_hierarchical_features.py \
--dataset-root-path `DATASET_ROOT` \
--dataset-name `DATASET_NAME` \
--feature-extractor-name `plip`|`quiltnet`|`conch` \
--low-mag 5x \
--high-mag 20x \
--max-patches 16
```

## 4. Verify Hierarchical Patch Feature Consistency

To verify that each 5x patch correctly aligns with its corresponding 20x patches, run the following consistency check:

```
python check_hierarchical_consistency.py \
--dataset-root-path `DATASET_ROOT` \
--dataset-name `DATASET_NAME` \
--feature-extractor-name `FEATURE_EXTRACTOR_NAME` \
```

## 5. Generate Hierarchical Text Prompts

HiVE-MIL leverages a LLM to generate hierarchical morphological descriptions for each class.

The prompt used for generation is as follows:

`Q: The task is to summarize the morphological features of the {dataset_name} dataset for the classes {class_1}, {class_2}, ..., {class_Y}. For each class, list four representative morphological features observed at 5× magnification, followed by three finer sub-features observed at 20× magnification for each.`


- The **first four entries** represent coarse-scale (5×) morphological descriptions.
- Each 5× description is **expanded into three fine-scale (20×) sub-descriptions**, reflecting more fine-grained morphological descriptions.

For reproducibility, we provide the generated hierarchical prompts for all datasets in the `text_prompt` directory

## 6. Split Dataset

For reproducibility, we provide the splits for all three datasets in the 4-shot, 8-shot, and 16-shot in the `splits` folder

## 7. Train 

```
python main.py \
--task `DATASET_NAME` \
--drop_out \
--early_stopping \
--few_shot_num `FEW_SHOT_NUM` \
--feature_extractor `FEATURE_EXTRACTOR_NAME`
```