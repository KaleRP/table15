import multiprocessing as mp
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from table15.models.model import Model
from table15.utils.data_tables import DataTables
from table15.utils.models_container import ModelsContainer

from . import magec_utils as mg


def generate_table_by_feature_type(data_tables: DataTables, models_container: ModelsContainer, 
                                   feature_type: str='numerical', use_multiprocessing: bool=True) -> pd.DataFrame:
    """For each feature type (numerical, binary, categorical, etc), get MAgEC perturbations by the following steps:
    1) Configs and pipeline parameters setup
    2) Choose multiprocessing if applicable. For now, we only support multiprocessing for sklean-based models.
    3) Perform feature perturbations
    4) Aggregate predictions outputs across population
    5) Generate table 1.5

    Args:
        data_tables (DataTables): Object that holds data information including train/test tables
        models_container (ModelsContainer): Contains trained models.
        feature_type (str, optional): One of the feature types, eg: numerical, binary, categorical. Defaults to 'numerical'.
        use_multiprocessing (bool, optional): Flag to use multiprocesing. Defaults to True.

    Returns:
        pd.DataFrame: Pandas Dataframe representing a visualized output of Table 1.5 for a feature_type
    """
    print(f'Generating Table1.5 for {feature_type} features')
    
    features = data_tables.get_features_by_type(feature_type)
    if features is None or len(features) == 0:
        return None, None

    data_configs = data_tables.data_configs
    if feature_type in ["numerical", "grouped"]:
        configs_key = f"{feature_type.upper()}_INTENSITIES"
        perturbation_intensities = data_configs.get_from_configs(configs_key, param_type="PERTURBATIONS", 
                                                                 default=[1., 0.5, 0.1])
    elif feature_type == 'binary':
        perturbation_intensities = [0, 1]
    elif feature_type == 'categorical':
        perturbation_intensities = [1]
        
    output_type = data_configs.get_from_configs("OUTPUT_TYPE", param_type="PERTURBATIONS")
    run_normalization = data_configs.get_from_configs("RUN_NORMALIZATION", param_type="PERTURBATIONS", default=False)
        
    perturbation_params = {
        "baselines": perturbation_intensities,
        "features": features,
        "feature_type": feature_type,
        "output_type": output_type,
        "run_normalization": run_normalization
    }

    if use_multiprocessing is True:
        baseline_runs = baseline_runs_via_multip(data_tables, models_container, perturbation_params)
    else:
        print('getting magecs for all models with single-processing ...')
        baseline_runs = generate_perturbation_predictions(
            data_tables, perturbation_params, models_container.models_dict)
    if isinstance(features[0], list):
        features = ["::".join(group) for group in data_tables.grouped_features]
    baseline_to_outputs_df = agg_models_per_baseline(baseline_runs, data_tables, models_container, features)

    df_out = visualize_output(baseline_to_outputs_df, perturbation_intensities, features, data_tables.test_stats_dict.get(feature_type))

    return df_out


def baseline_runs_via_multip(data_tables: DataTables, models_container: ModelsContainer, 
                             perturbation_params: Dict[str, Any]) -> Dict[float, List[pd.DataFrame]]:
    """Sets up multiprocessing of each perturbation intensity and model combination.
        Note, only sklean models are supported for multiprocessing at this time.

    Args:
        data_tables (DataTables): Object that holds data information including train/test tables
        models_container (ModelsContainer): Contains trained models.
        perturbation_params (Dict[str, Any]): Helper dictionary to pass parameters for perturbations.

    Returns:
        Dict[float, List[pd.DataFrame]]: Dictionary of key: perturbation intensity, 
            and value: Dict of model and supporting dataframes,
    """
    # Flag for single-process models
    has_tf_models = False
    if 'mlp' in models_container.models_dict:
        has_tf_models = True

    mp_models_dict = models_container.models_dict.copy()
    if has_tf_models:
        tf_models_list = ['mlp']
        # if use_ensemble is True:
        #     tf_models_list.append('ensemble')
        tf_models_dict = {tf_model: models_container.models_dict[tf_model] for tf_model in tf_models_list}
        for tf_model in tf_models_list:
            del mp_models_dict[tf_model]

    with mp.Manager() as manager:
        print('getting magecs for non-TF models via multiprocessing...')
        baseline_runs = generate_perturbation_predictions(
            data_tables, perturbation_params, mp_models_dict, mp_manager=manager)
        print('Done multiprocessing')
    
    if has_tf_models:
        print('getting magecs for TF models with single-processing ...')
        tf_baseline_runs = generate_perturbation_predictions(
            data_tables, perturbation_params, tf_models_dict, mp_manager=None)
        baselines = perturbation_params["baselines"]
        baseline_runs = combine_baseline_runs(baseline_runs, tf_baseline_runs, baselines)
    
    return baseline_runs


def generate_perturbation_predictions(data_tables: DataTables, perturbation_params: Dict[str, Any], 
                                      models_dict: Dict[str, Model], mp_manager: mp.Manager=None
                                      ) -> Dict[float, List[pd.DataFrame]]:
    """For either single or multiprocess, perform the perturbations and use `key` to easily reference each run.

    Args:
        data_tables (DataTables): Object that holds data information including train/test tables
        perturbation_params (Dict[str, Any]): Helper dictionary to pass parameters for perturbations
        models_dict (Dict[str, Model]): Holds all trained models
        mp_manager (mp.Manager, optional): Multiprocessor object. Defaults to None.

    Returns:
        Dict[str,  Dict[float, List[pd.DataFrame]]]: Dict to hold all the perturbation runs for a single feature type.
    """
    is_multi_process = False
    run_dfs = dict()
    if mp_manager is not None:
        is_multi_process = True
        run_dfs = mp_manager.dict()
        processes = []
    
    keys = []
    baselines = perturbation_params["baselines"]
    for baseline in baselines:
        perturbation_params["baseline"] = baseline
        for model_name in models_dict.keys():
            perturbation_params["model_name"] = model_name
            key = model_name + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model_name + '_0'
            keys.append(key)
            clf = models_dict[model_name]
            if is_multi_process is False:
                if model_name in ['lstm']:
                    clf = clf.model
                run_dfs[key] = run_magecs_single_process(clf, data_tables, key, perturbation_params.copy())
            elif is_multi_process is True:
                p = mp.Process(name=key, target=run_magecs_multiprocess, 
                    args=(run_dfs, clf, data_tables, perturbation_params.copy()))
                processes.append(p)
        
    if is_multi_process:
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    baseline_runs = store_run_dfs_by_baseline(run_dfs, keys)
    return baseline_runs


def run_magecs_single_process(clf: Model, data_tables: DataTables, key: str, 
                              perturbation_params: Dict[str, Any]) -> pd.DataFrame:
    """Run MAgEC generation through single process for TensorFlow based models or for debugging.

    Args:
        clf (Model):  Implemented model object
        data_tables (DataTables): Object that holds data information including train/test tables
        key (str): Used to easily reference each run.
        perturbation_params (Dict[str, Any]): Helper dictionary to pass parameters for perturbations

    Returns:
        _type_: A Pandas dataframe containing MAgEC perturbations
    """
    print('Starting single-process:', key)
    magecs = run_magecs(key, clf, data_tables, perturbation_params)
    return magecs
    

def run_magecs_multiprocess(return_dict: Dict[str, pd.DataFrame], clf: Model, data_tables: DataTables, 
                            perturbation_params: Dict[str, Any]) -> None:
    """Run MAgEC generation through multiprocessing

    Args:
        return_dict (Dict[str, pd.DataFrame]): Output dictionary
        clf (Model): Implemented model object
        data_tables (DataTables): Object that holds data information including train/test tables
        perturbation_params (Dict[str, Any]): Helper dictionary to pass parameters for perturbations
    """
    p_name = mp.current_process().name
    print('Starting multi-process:', p_name)
    magecs = run_magecs(p_name, clf, data_tables, perturbation_params)
    return_dict[p_name] = magecs
    

def run_magecs(name: str, model: Model, data_tables: DataTables, perturbation_params: Dict[str, Any]) -> pd.DataFrame:
    """Calls the MAgEC feature perturbation method.
    Note, the normalization step generally produces values that are hard to interpret and likely should not be run.

    Args:
        name (str): Unique key name for internal tracking
        model (Model): Trained model used for predictions on test set
        data_tables (DataTables): Object that holds data information including train/test tables
        perturbation_params (Dict[str, Any]): Helper dictionary to pass parameters for perturbations

    Returns:
        pd.DataFrame: Generated outputs based on MAgECS in dataframe
    """
    magecs = mg.case_magecs(model, data_tables.x_test, perturbation_params, data_tables.setted_numerical_values)
    print('Magecs for {} computed...'.format(name))
    if perturbation_params['run_normalization'] == True:
        magecs = mg.normalize_magecs(magecs, features=perturbation_params["features"], model_name=perturbation_params["model_name"])
        print('Magecs for {} normalized...'.format(name))
    magecs = magecs.merge(data_tables.Y_test, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', name)
    return magecs


def combine_baseline_runs(main_dict: Dict[float, List[pd.DataFrame]], 
                          to_combine_dict: Dict[float, List[pd.DataFrame]], 
                          baselines: List[float]) -> Dict[float, List[pd.DataFrame]]:
    """Used to combines baseline runs from TF models and non-TF models.

    Args:
        main_dict (Dict[float, List[pd.DataFrame]]): Dict with calculated MAgEC values
        to_combine_dict (Dict[float, List[pd.DataFrame]]): Dict with calculated MAgEC values
        baselines (List[float]): List of perturbation intensities

    Returns:
        Dict[float, List[pd.DataFrame]]: Combined dict
    """
    for baseline in baselines:
        main_dict[baseline].extend(to_combine_dict[baseline])
    return main_dict


def agg_models_per_baseline(baseline_runs: Dict[float, List[pd.DataFrame]], data_tables: DataTables, models_container: ModelsContainer, 
                              features: List[str], use_rank: bool=False) -> Dict[float, List[pd.DataFrame]]:
    """Aggregate outputs of each baseline run's outputs. Scores can be generated through a top-n ranking algorithm (deprecated) or simply 
        by aggregating outputs by mean.

    Args:
        baseline_runs (Dict[float, List[pd.DataFrame]]]): Dict of MAgEC outputs
        data_tables (DataTables): Object that holds data information including train/test tables
        models_container (ModelsContainer): Contains trained models.
        features (List[str]): List of features
        use_rank (bool, optional): Warning: this uses a deprecated method that takes top-N ranked outputs. Defaults to False.

    Returns:
        Dict[float, List[pd.DataFrame]]: Dict of dataframes with outputs aggregated from all models for each baseline
    """
    baseline_to_outputs_df = {}
    model_names = list(models_container.models_dict.keys())
    for baseline, model_runs in baseline_runs.items():
        model_runs_per_baseline = mg.magec_models(*model_runs,
                            Xdata=data_tables.x_test,
                            Ydata=data_tables.Y_test,
                            features=features)
        if use_rank is False:
            outputs_df = aggregate_outputs_of_models(model_runs_per_baseline, model_names, features)
        else:
            baseline_ranked_df = mg.magec_rank(model_runs_per_baseline, 
                                            rank=len(features), features=features, models=model_names)
            outputs_df = agg_outputs(baseline_ranked_df, models_container.model_feat_imp_dict, model_names)
        baseline_to_outputs_df[baseline] = outputs_df
    return baseline_to_outputs_df


def aggregate_outputs_of_models(model_runs_per_baseline: pd.DataFrame, model_names: List[str], features: List[str]) -> pd.DataFrame:
    """Aggregate the outputs of each model for each feature for all cases by taking their average values.

    Args:
        model_runs_per_baseline (pd.DataFrame): Dataframe of calculated output values for each feature, for each case
        model_names (List[str]): List of model names
        features (List[str]): List of features

    Returns:
        pd.DataFrame: Dataframe with aggregated output values
    """
    feats_to_agg_cols = [[mg.create_magec_col(m, f) for m in model_names] for f in features]
    agg_series_list = []
    for cols, feat in zip(feats_to_agg_cols, features):
        agg_series = pd.Series(model_runs_per_baseline[cols].mean(axis=1), name=feat)
        agg_series_list.append(agg_series)
    return pd.concat(agg_series_list, axis=1)


def agg_outputs(ranked_df, model_feat_imp_dict, models):
    """Warning: deprecated.
    """
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        outputs = mg.magec_scores(magecs_feats, row, model_feat_imp_dict, use_weights=False)
        out.append(outputs)
    
    return pd.DataFrame.from_records(out)


def store_run_dfs_by_baseline(run_dfs: Dict[str, pd.DataFrame], keys: str) -> Dict[float, List[pd.DataFrame]]:
    """Use key to extract MAgEC outputs and store these into Dict of baselines

    Args:
        run_dfs (Dict[str, pd.DataFrame]): Dict of internal key to generated MAgEC output
        key (str): Used to easily reference each run.

    Returns:
        Dict[float, List[pd.DataFrame]]: Container of all outputs
    """
    baseline_runs = defaultdict(list)
    for key in keys:
        baseline = key.split('_')[1]
        if baseline[0] == 'p':
            baseline = int(baseline[1:]) / 100
        else:
            baseline = int(baseline)
        baseline_runs[baseline].append(run_dfs[key])
    return baseline_runs


def visualize_output(baseline_to_outputs_df: Dict[float, List[pd.DataFrame]], baselines: List[float], features: List[str], 
                     test_stats_dict_feature_type: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """Take MAgEC outputs and convert to a visualized Table 1.5

    Args:
        baseline_to_outputs_df (Dict[float, List[pd.DataFrame]]): Dict of dataframes with outputs aggregated from all models for each baseline
        baselines (List[float]): List of perturbation intensities
        features (List[str]): List of features
        test_stats_dict_feature_type (Dict[str, Dict[str, pd.Series]]): Useful statistics from x_test

    Returns:
        pd.DataFrame: Table 1.5 for a feature_type containing both aggregated outputs and useful statistics
    """
    output = {}
    for baseline in baselines:
        df_out = pd.DataFrame.from_records(baseline_to_outputs_df[baseline])
        output[baseline] = get_string_repr(df_out, features)
    
    # # TODO: fix baselines upstream  to handle None as 0
    # formatted_baselines = baselines.copy()

    df_out =  produce_output_df(output, features, baselines, test_stats_dict_feature_type)
    return df_out


def get_string_repr(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Produce aggregated mean of MAgEC outputs for each feature, and other stats to display as string

    Args:
        df (pd.DataFrame): Dataframe of MAgEC outputs
        features (List[str]): List of features

    Returns:
        List[str]: List of string representations to display in Table 1.5
    """
    base_strings = []
    if not df.empty:
        for feat in features:
            mean = round(df[feat].mean(), 4)
            # std = round(df[feat].std(), 4)
            sem = round(df[feat].sem(), 4)
            # string_repr = f'{mean} +/- {std}'
            string_repr = f'{mean:.3f} ({sem:.3f})'
            base_strings.append(string_repr)
    return base_strings


def produce_output_df(output: Dict[float, pd.DataFrame], features: List[str], baselines: List[float], 
                      test_stats_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Produce final outputs

    Args:
        output (Dict[float, pd.DataFrame]): Aggregated outputs of each baseline
        features (List[str]): List of features
        baselines (List[float]): List of baselines
        test_stats_dict (Dict[str, pd.Series]): Final Table 1.5 for a feature_type

    Returns:
        pd.DataFrame: _description_
    """
    df_out = pd.DataFrame.from_records(output)
    df_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    # df_out = df_out.rename(columns={'0': 'full'})
    df_out = df_out[cols]
    
    if test_stats_dict is not None:
        for stat, stats_series in test_stats_dict.items():
            df_out[stat] = stats_series.values
    
    return df_out
