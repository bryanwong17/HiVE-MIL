def get_default_class_labels():
    class_names = ['0', '1']
    return class_names

def get_tcga_nsclc_class_labels():
    class_names = ['LUAD', 'LUSC']
    return class_names

def get_tcga_brca_class_labels():
    class_names = ['IDC', 'ILC']
    return class_names

def get_tcga_rcc_class_labels():
    class_names = ['CCRCC', 'CHRCC', 'PRCC']
    return class_names

def get_class_names(dataset_name):
    if dataset_name == 'tcga_nsclc':
        class_names = get_tcga_nsclc_class_labels()
    elif dataset_name == 'tcga_brca':
        class_names = get_tcga_brca_class_labels()
    elif dataset_name == 'tcga_rcc':
        class_names = get_tcga_rcc_class_labels()
    elif dataset_name is None:
        print('Not specify dataset, use default dataset with label 0, 1 instead')
        class_names = get_default_class_labels()
    else:
        raise NotImplementedError

    return class_names
