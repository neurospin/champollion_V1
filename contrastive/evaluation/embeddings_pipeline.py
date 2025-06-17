import os
import yaml
import json
import omegaconf
import inspect

from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers
from utils_pipelines import get_save_folder_name, change_config_datasets,\
                            change_config_label, change_config_dataset_localization

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, dataset_localization,
                      datasets_root, datasets, idx_region_evaluation,
                      label, folder_name, classifier_name='svm',
                      epoch=None, split=None, cv=5,
                      splits_basedir=None, verbose=False):
    """Loads the associated config of the given model and changes what has to be done,
    mainly the datasets, the classifier type and a few other keywords.
    
    Arguments:
        - sub_dir: str. Path to the directory containing the saved model.
        - datasets: list of str. List of the datasets to be used for the results generation.
        - label: str. Name of the label to be used for evaluation.
        - folder_name: str. Name of the directory where to store both embeddings and aucs.
        - classifier_name: str. Should correspond to a classifier yaml file's name 
        (currently either 'svm' or 'neural_network').
        - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
        - verbose: bool. Verbose.
        
    Output:
        - cfg: the config as an omegaconf object."""

    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the datasets
    change_config_datasets(cfg, datasets, datasets_root)
    # replace the label
    change_config_label(cfg, label)
    # replace the dataset localizatyion
    change_config_dataset_localization(cfg, dataset_localization)

    # get the right classifiers parameters
    with open(os.getcwd() + f'/configs/classifier/{classifier_name}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    # replace the possibly incorrect config parameters
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = \
        sub_dir + f"/{folder_name}_embeddings"
    cfg.training_embeddings = \
        sub_dir + f"/{folder_name}_embeddings"
    cfg.apply_transformations = False
    cfg.multiregion_single_encoder = False
    cfg.load_sparse = False

    # add epoch to config if specified
    if epoch is not None:
        cfg.epoch = epoch
    # add splitting strategy to config
    cfg.split = split
    if split=='custom':
        cfg.splits_basedir=splits_basedir
    elif split=='random':
        cfg.cv=cv

    # in multi head case
    if idx_region_evaluation is not None:
        cfg.idx_region_evaluation=idx_region_evaluation

    # change config partition to avoid errors
    cfg.partition = [0.9,0.1]

    return cfg


def is_it_a_file(sub_dir):
    if os.path.isdir(sub_dir):
        return False
    else:
        print(f"{sub_dir} is a file. Continue.")
        return True
    

def is_folder_a_model(sub_dir):
    if os.path.exists(sub_dir+'/.hydra/config.yaml'):
        return True
    else:
        print(f"\n{sub_dir} not associated to a model. Continue")
        return False
    

def is_folder_accepted_model(sub_dir):
    if '#' in sub_dir:
        print(
            "Model with an incompatible structure "
            "with the current one, because there is # in the name."
            "Pass."
            )
        return False
    else:
        return True


def get_model_folder_name(epoch, folder_name):
    if epoch is not None:
        f_name = folder_name + f'_epoch{epoch}'
    else:
        f_name = folder_name
    return f_name


def print_config(cfg, verbose):
    if verbose:
        print("CONFIG FILE", type(cfg))
        print(json.dumps(omegaconf.OmegaConf.to_container(
            cfg, resolve=True), indent=4, sort_keys=True))


def save_classifier_config(cfg, sub_dir):
    # save the modified classifier config next to the real one
    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') \
            as file:
        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)


def reload_classifier_config(sub_dir):
    # reload config for train_classifiers to work properly
    cfg = omegaconf.OmegaConf.load(
        sub_dir+'/.hydra/config_classifiers.yaml')
    return cfg


def check_if_compute_embedding(sub_dir, f_name, overwrite, embeddings, idx):
    if (
        os.path.exists(sub_dir + f"/{f_name}_embeddings")
        and (not overwrite)
    ):
        print(f"Model {f_name} already treated "
            "(existing folder with embeddings). "
            "Set overwrite to True if you still want "
            "to compute them.")
        do_we_compute_embeddings = False
        valid_path=True # assume that the embeddings exist
    else:
        # apply the functions
        if embeddings and idx==0:
            do_we_compute_embeddings = True
            valid_path = False # will be set during embedding computation
        elif not embeddings:
            do_we_compute_embeddings = False
            valid_path=True # assume that the embeddings exist 
    return do_we_compute_embeddings, valid_path


def do_we_classify(valid_path, embeddings_only):
    if valid_path and not embeddings_only:
        return True
    elif not valid_path:
        print('Invalid epoch number, skipped')
        return False
    else:
        return False 


# main function
# creates embeddings and train classifiers for all models contained in folder
@ignore_warnings(category=ConvergenceWarning)
def embeddings_pipeline(dir_path, dataset_localization,
                        datasets_root, datasets, idx_region_evaluation, labels,
                        short_name=None, classifier_name='svm',
                        overwrite=False, embeddings=True, embeddings_only=False,
                        use_best_model=False, subsets=['full'],
                        epochs=None, split='random', cv=5, splits_basedir=None, verbose=False):
    """Pipeline to generate automatically the embeddings and compute the associated AUCs 
    for all the models contained in a given directory. All the AUCs are computed with 
    5-folds cross validation .

    Arguments:
        - dir_path: str. Path where the models are stored and where is applied 
        recursively the process.
        - dataset_localization: gives position of dataset
        - datasets: list of str. Datasets the embeddings are generated from.
        - labels: str list. Names of the labels to be used for evaluation.
        - short_name: str or None. Name of the directory where to store both embeddings 
        and aucs. If None, use datasets to generate the folder name.
        - classifier_name: str. Parameter to select the desired classifer type
        (currently neural_network or svm).
        - overwrite: bool. Redo the process on models where embeddings already exist.
        - embeddings: bool. Compute the embeddings, or use the ones previously computed.
        - use_best_model: bool. Use the best model saved during to generate embeddings. 
        The 'normal' model is always used, the best is only added.
        - subsets: list of subsets you want the SVM to learn on. Set to ['full'] if you
        want to learn on all subjects in one go.
        - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
        - verbose: bool. Verbose.
    """

    print("/!\\ Convergence warnings are disabled")

    # Gets function parameters to call it recursively with same parameters
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    args_function = {i: values[i] for i in args}     

    # walks recursively through the subfolders
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if is_it_a_file(sub_dir):
            pass
        elif not is_folder_a_model(sub_dir):
            args_function["dir_path"] = sub_dir
            embeddings_pipeline(**args_function)
        elif not is_folder_accepted_model(sub_dir):
            pass
        else:
            print("\nTreating", sub_dir)

            folder_name = get_save_folder_name(datasets=datasets,
                                               short_name=short_name+'_'+split)

            print("Start computing")

            # Loops over labels
            for idx, label in enumerate(labels):

                # Loops over epochs if requested
                for epoch in epochs:
                    f_name = get_model_folder_name(epoch, folder_name)

                    # Takes the model configuration
                    # And updates it with input parameters
                    cfg = preprocess_config(
                        sub_dir,
                        dataset_localization=dataset_localization,
                        datasets_root=datasets_root,
                        datasets=datasets,
                        idx_region_evaluation=idx_region_evaluation,
                        label=label,
                        folder_name=f_name,
                        classifier_name=classifier_name,
                        epoch=epoch, split=split, cv=cv,
                        splits_basedir=splits_basedir)
                    
                    print_config(cfg, verbose)
                    save_classifier_config(cfg, sub_dir)

                    ####################
                    # Compute embeddings
                    ####################
                    do_we_compute_embeddings, valid_path =\
                        check_if_compute_embedding(sub_dir, f_name, overwrite,
                                                   embeddings, idx)
                    if do_we_compute_embeddings == True:
                        valid_path = compute_embeddings(cfg, subsets=subsets)
                    
                    ####################
                    # Compute Classifier
                    ####################
                    cfg = reload_classifier_config(sub_dir)
                    if do_we_classify(valid_path, embeddings_only):
                        train_classifiers(cfg, subsets=subsets)


                    #######################################
                    # compute embeddings for the best model
                    #######################################
                    if (use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt')):
                        print("\nCOMPUTE AGAIN WITH THE BEST MODEL\n")
                        # apply the functions
                        cfg = omegaconf.OmegaConf.load(
                            sub_dir+'/.hydra/config_classifiers.yaml')
                        cfg.use_best_model = True
                        if embeddings and idx==0:
                            _ = compute_embeddings(cfg, subsets=subsets)
                        # reload config for train_classifiers to work properly
                        cfg = omegaconf.OmegaConf.load(
                            sub_dir+'/.hydra/config_classifiers.yaml')
                        cfg.use_best_model = True
                        cfg.training_embeddings = cfg.embeddings_save_path + \
                            '_best_model'
                        cfg.embeddings_save_path = \
                            cfg.embeddings_save_path + '_best_model'
                        train_classifiers(cfg, subsets=subsets)



if __name__ == "__main__":

    # Cadasil, without supervision
    embeddings_pipeline("/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation",
        dataset_localization="neurospin",
        datasets_root="julien/NR2F1_GD_2025/array_load",
        datasets=["toto"],
        idx_region_evaluation=None,
        labels=["Sex"],
        classifier_name='logistic',
        short_name='NR2F1_GD_2025', overwrite=True, embeddings=True, embeddings_only=True, use_best_model=False,
        subsets=['full'], epochs=[None], split='random', cv=5,
        splits_basedir='',
        verbose=False) 


    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/2024-06-21",
    #                     dataset_localization="neurospin",
    #                     datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
    #                     labels=['Left_OFC'],
    #                     short_name='troiani', overwrite=True, embeddings=True,
    #                     embeddings_only=False, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='custom', cv=3,
    #                     splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
    #                     verbose=False)

    """
    embeddings_pipeline("/volatile/jl277509/Runs/02_STS_babies/Output/multiple_regions",
                        dataset_localization="neurospin",
                        datasets=["julien/multiple_regions_UKB_2000subs"],
                        labels=['region'],
                        classifier_name='logistic',
                        short_name='ukb', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['train_val'], epochs=range(0,20,10), split='random', cv=5,
                        splits_basedir='',
                        verbose=False)
    
    """
    
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_full/8_trimdepth_translation_3/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_full/9_trimextremities_translation_3/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    """

    """2                 
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_combinations/combinations_with_trim/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    """
                        

    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/10_cutin/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/11_cutout/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    """
    
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/9_trimextremities/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/10_cutin/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/11_cutout/SOr_left_UKB40",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left/train_val_split_',
                        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/orbital_kernel5",
                    dataset_localization="neurospin",
                    datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                    idx_region_evaluation = None,
                    labels=['Left_OFC'],
                    classifier_name='logistic',
                    short_name='troiani', overwrite=True, embeddings=True,
                    embeddings_only=False, use_best_model=False,
                    subsets=['full'], epochs=[None], split='custom', cv=3,
                    splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
                    verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/orbital_no_domain_specific_augm",
                    dataset_localization="neurospin",
                    datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                    idx_region_evaluation = None,
                    labels=['Left_OFC'],
                    classifier_name='logistic',
                    short_name='troiani', overwrite=True, embeddings=True,
                    embeddings_only=False, use_best_model=False,
                    subsets=['full'], epochs=[None], split='custom', cv=3,
                    splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
                    verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/4_regions_pretrain",
                    dataset_localization="neurospin",
                    datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
                    idx_region_evaluation = None,
                    labels=['Left_OFC'],
                    classifier_name='logistic',
                    short_name='troiani', overwrite=True, embeddings=True,
                    embeddings_only=False, use_best_model=False,
                    subsets=['full'], epochs=[None], split='custom', cv=3,
                    splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
                    verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/10_regions_flip_acc3",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/orbital_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_OFC'],
                        classifier_name='logistic',
                        short_name='troiani', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
                        verbose=False)
    """

    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/orbital_extremities_pepper",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/training/orbital_left_UKB"],
                        idx_region_evaluation = None,
                        labels=['isOld'],
                        classifier_name='logistic',
                        short_name='ukb', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='random', cv=3,
                        splits_basedir='',
                        verbose=False)
    """
    
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_combinations/combinations_with_trim/LARGE_CINGULATE_right_UKB40",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/LARGE_CINGULATE_right_ACCpatterns_custom"],
        idx_region_evaluation = None,
        labels=['Right_PCS'],
        classifier_name='logistic',
        short_name='ACC', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/splits/Right/train_val_split_',
        verbose=False)
    """
    """
    embeddings_pipeline('/neurospin/dico/jlaval/Output/CINGULATE_40k',
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/training/cingulate_40k_right"],
        idx_region_evaluation = None,
        labels=['isOld'],
        classifier_name='logistic',
        short_name='UKB', overwrite=True, embeddings=True, embeddings_only=True, use_best_model=False,
        subsets=['full'], epochs=[None], split='random', cv=3,
        splits_basedir=None,
        verbose=False)
    """

    """
    embeddings_pipeline("/neurospin/dico/data/deep_folding/current/models/Champollion_V0/CINGULATE_left/2024-07-15",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/cingulate_left_CCD"],
        idx_region_evaluation = None,
        labels=['Left_PCS'],
        classifier_name='logistic',
        short_name='CCD', overwrite=True, embeddings=True, embeddings_only=True, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/ACCpatterns_subjects_train_split_',
        verbose=False)
    """

    # custom cv (80%)
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_combinations/combinations_with_trim/FIP_right_UKB40/",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/Right/train_val_split_',
                        verbose=False)
    """

    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/9_trimextremities/FIP_right_UKB40/",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/Right/train_val_split_',
                        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/10_cutin/FIP_right_UKB40/",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/Right/train_val_split_',
                        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/11_cutout/FIP_right_UKB40/",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/Right/train_val_split_',
                        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/FIP_cutin_and_trim",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/split_',
                        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ABLATION_FIP",
                        dataset_localization="neurospin",
                        datasets=["julien/UKB40/sparse_load/FIP_right_UKB40_sparse_load"],
                        idx_region_evaluation = None,
                        labels=['Age'],
                        classifier_name='logistic',
                        short_name='42433_ukb_FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='random', cv=5,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/split_',
                        verbose=False)
    """
    
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/FIP_kernel5",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/split_',
                        verbose=False)
    """
    # full FIP dataset (390)
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/V1_FIP_right",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_hcp_400_cv"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_right', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split=None, cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/split_',
                        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/5_models_FIP_right_3_layer_proj",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_right_htp"],
                        idx_region_evaluation = None,
                        labels=['Right_FIP'],
                        classifier_name='logistic',
                        short_name='htp', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['train_val'], epochs=[None], split='random', cv=3,
                        splits_basedir=None,
                        verbose=False)
    """
    
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/FIP_left_ConvNet_v1",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_left_hcp_custom"],
                        idx_region_evaluation = None,
                        labels=['Left_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_left', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/split_',
                        verbose=False)
    """
    # full 390 subs
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/5_models_FIP_left_3_layer_proj",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/FIP_left_hcp_400_cv"],
                        idx_region_evaluation = None,
                        labels=['Left_FIP'],
                        classifier_name='logistic',
                        short_name='FIP_left', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='custom', cv=3,
                        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/split_',
                        verbose=False)
    """

    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/5_models_FIP_left_3_layer_proj",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/training/FIP_40k_left"],
                        idx_region_evaluation = None,
                        labels=['isOld'],
                        classifier_name='logistic',
                        short_name='ukb40', overwrite=True, embeddings=True,
                        embeddings_only=True, use_best_model=False,
                        subsets=['full'], epochs=[None], split='random', cv=3,
                        splits_basedir='',
                        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/FIP_right_ConvNet_v1",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/training/FIP_40k_right"],
                        idx_region_evaluation = None,
                        labels=['isOld'],
                        classifier_name='logistic',
                        short_name='ukb40', overwrite=True, embeddings=True,
                        embeddings_only=True, use_best_model=False,
                        subsets=['full'], epochs=[None], split='random', cv=3,
                        splits_basedir='',
                        verbose=False)
    """

    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_combinations/combinations_with_trim/SC-sylv_left_UKB40",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_left_isomap"],
        idx_region_evaluation=None,
        labels=[f'Isomap_central_left_dim{k}' for k in range(1,7)],
        classifier_name='logistic',
        short_name='hcp_isomap', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/Isomap/splits/train_val_split_',
        verbose=False)
    """
        
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/9_trimextremities/SC-sylv_left_UKB40",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_left_isomap"],
        idx_region_evaluation=None,
        labels=[f'Isomap_central_left_dim{k}' for k in range(1,7)],
        classifier_name='logistic',
        short_name='hcp_isomap', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/Isomap/splits/train_val_split_',
        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/10_cutin/SC-sylv_left_UKB40",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_left_isomap"],
        idx_region_evaluation=None,
        labels=[f'Isomap_central_left_dim{k}' for k in range(1,7)],
        classifier_name='logistic',
        short_name='hcp_isomap', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/Isomap/splits/train_val_split_',
        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/11_cutout/SC-sylv_left_UKB40",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_left_isomap"],
        idx_region_evaluation=None,
        labels=[f'Isomap_central_left_dim{k}' for k in range(1,7)],
        classifier_name='logistic',
        short_name='hcp_isomap', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/Isomap/splits/train_val_split_',
        verbose=False)
    """

    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/SC-sylv_12-16",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_left_ukb_morpho"],
        idx_region_evaluation=None,
        labels=['Mean_depth_talairach'],
        short_name='troiani', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='random', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/SC-sylv_left",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_left_hcp_train_test"],
        idx_region_evaluation=None,
        labels=[f'Isomap_central_left_dim{k}' for k in range(1,7)],
        short_name='troiani', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='train_test', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    """
    
    ## SC-sylv_right UKB
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/SC-sylv_right_V1",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/training/SC-sylv_right_40k"],
        idx_region_evaluation=None,
        labels=['Sex'],
        short_name='ukb40', overwrite=True, embeddings=True, embeddings_only=True, use_best_model=False,
        subsets=['full'], epochs=[None], split='random', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/2025-03-07",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_right_interruption_UKB40"],
        idx_region_evaluation=None,
        labels=['Interruption_SC_right'],
        short_name='ukb40_interrupted', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['train_val'], epochs=[None], split='random', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    """
    
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/5_trimextremities_SC_right/1_all_augmentations/2025-02-26",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_right_interruption_UKB40"],
        idx_region_evaluation=None,
        labels=['Interruption_SC_right'],
        short_name='ukb40_interrupted', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['train_val'], epochs=[None], split='random', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/5_trimextremities_SC_right/2_no_trimextremities/2025-02-26",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_right_interruption_UKB40"],
        idx_region_evaluation=None,
        labels=['Interruption_SC_right'],
        short_name='ukb40_interrupted', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['train_val'], epochs=[None], split='random', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/5_trimextremities_SC_right/3_all_trimextremities_p80/2025-02-26",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/SC-sylv_right_interruption_UKB40"],
        idx_region_evaluation=None,
        labels=['Interruption_SC_right'],
        short_name='ukb40_interrupted', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['train_val'], epochs=[None], split='random', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
    """
    ## imagen
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/4_regions_pretrain",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/deMatos_polar_left_imagen_random"],
                        idx_region_evaluation = None,
                        labels=['Left_Interrup_CS_OTS', 'Left_Interrup_RS_CS', 'Left_Interrup_RS_OTS'],
                        classifier_name='logistic',
                        short_name='imagen', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='random', cv=5,
                        splits_basedir='',
                        verbose=False)
    """
    """
    embeddings_pipeline("/neurospin/dico/jlaval/Output/imagen_right",
                        dataset_localization="neurospin",
                        datasets=["julien/MICCAI_2024/evaluation/deMatos_right_imagen_random"],
                        idx_region_evaluation = None,
                        labels=['Right_Interrup_CS_OTS', 'Right_Interrup_RS_CS', 'Right_Interrup_RS_OTS'],
                        classifier_name='logistic',
                        short_name='imagen', overwrite=True, embeddings=True,
                        embeddings_only=False, use_best_model=False,
                        subsets=['full'], epochs=[None], split='random', cv=5,
                        splits_basedir='',
                        verbose=False)
    """

    """
    # Isomap cingulate    
    embeddings_pipeline("/neurospin/dico/jlaval/Output/ablation_2_models_combinations/combinations_with_trim/LARGE_CINGULATE_right_UKB40",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/LARGE_CINGULATE_right_isomap"],
        idx_region_evaluation=None,
        labels=[f'Isomap_cingulate_right_dim{k}' for k in range(1,7)],
        classifier_name='logistic',
        short_name='hcp_isomap', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=5,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/hcp/Isomap/splits/train_val_split_',
        verbose=False)
    """

    

    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/ORBITAL_BT",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/schiz_extended/ORBITAL_left"],
    #                     labels=['diagnosis'],
    #                     short_name='schiz_extended', overwrite=True, embeddings=True,
    #                     embeddings_only=True, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)
    
    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/ORBITAL_BT",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/schiz_extended/ORBITAL_left"],
    #                     labels=['diagnosis'],
    #                     short_name='schiz_extended', overwrite=True, embeddings=True,
    #                     embeddings_only=True, use_best_model=True,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)

    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/2024-06-06_pretraining",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/schiz_extended/SC_SPeC_left_female",
    #                               "with_reskel_distbottom/2mm/schiz_extended/SC_SPeC_right_female"],
    #                     labels=['diagnosis'],
    #                     short_name='schiz_extended', overwrite=True, embeddings=True,
    #                     embeddings_only=False, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)
    
    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/2024-06-06_pretraining",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/UKB/SC_SPeC_left",
    #                               "with_reskel_distbottom/2mm/UKB/SC_SPeC_right"],
    #                     labels=['diagnosis'],
    #                     short_name='ukb', overwrite=True, embeddings=True,
    #                     embeddings_only=True, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)

"""
    embeddings_pipeline("/volatile/jl277509/Runs/02_STS_babies/Output/sparse_multiregion_test",
        dataset_localization="neurospin",
        datasets=["julien/MICCAI_2024/evaluation/cingulate_right_ACCpatterns_custom"],
        labels=['Right_PCS'],
        short_name='ACC', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['test'], epochs=[None], split='custom', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/ACCpatterns_subjects_train_split_',
        verbose=False)
"""        


"""OFC
        datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
        labels=['Left_OFC'],
        short_name='troiani', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
"""

"""STS Preterm
        datasets=["local_julien/1-5mm/STs_babies_dHCP_374_subjects_right_1-5mm"],
        labels=['Preterm_23-28_vs_fullterm'],
        short_name='dHCP', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=range(0,250,10), split='random', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
"""


#datasets=["local_julien/old/STs_dHCP_374_subjects"]
#split='random', 'custom'
#epochs=[None], range(0, 250, 10)
#subset=['full'], ['train_val']
#labels=['Preterm_28', 'Preterm_32', 'Preterm_37']
#labels=['Preterm_23-28_vs_fullterm']
#short_name='UKB_5percent'
#datasets=["local_julien/1-5mm/STs_babies_UKB_right_5percent_1-5mm"]
#datasets=["local_julien/1-5mm/STs_babies_dHCP_374_subjects_right_1-5mm"]

#    embeddings_pipeline("/volatile/jl277509/Runs/02_STS_babies/Program/Output/old_best/",
#        datasets=["local_julien/cingulate_UKB_right_5percent"],
#        labels=['Age', 'Age_64', 'Sex'],
#        short_name='UKB_5percent', overwrite=True, embeddings=True, use_best_model=False,
#        subsets=['train_val'], epochs=[None], verbose=False)


#datasets=["local_julien/cingulate_UKB_right"]
#datasets=["local_julien/cingulate_ACCpatterns_1
#label='Age_64'
#label='Right_PCS'
#short_name='UKB_1'
