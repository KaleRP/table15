import multiprocessing as mp
import os
import time
import warnings
from collections import defaultdict
from email.mime import base
from multiprocessing import set_start_method

import pandas as pd

from . import magec_utils as mg
from . import pima_utils as pm
from . import pipeline_utils as plutils


def run(configs_path='../configs/pima_diabetes.yaml'):
    if not os.path.isabs(configs_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        configs_path = os.path.join(script_dir, configs_path)
    print(configs_path)

    warnings.filterwarnings('ignore')
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    
    # TODO: adjust spawn method to start WITH multiprocessing. Most likely with mp.Pool()

    print('This is Version: 0.0.10')

    configs = plutils.yaml_parser(configs_path)
    baselines = plutils.get_from_configs(configs, 'BASELINES', param_type='CONFIGS')
    models = plutils.get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    policy = plutils.get_from_configs(configs, 'POLICY', param_type='CONFIGS')
    use_ensemble = plutils.get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')

    df, features, x_train_p, x_validation_p, y_train_p, y_validation_p = plutils.generate_data(configs)

    # Train models
    print('Training models ...')
    models_dict = plutils.train_models(x_train_p, y_train_p, models, use_ensemble=use_ensemble)
    print('Finished training models')

    # Flag for single-process models
    has_tf_models = False
    if 'mlp' in models_dict:
        has_tf_models = True

    mp_models_dict = models_dict.copy()
    if has_tf_models:
        tf_models_list = ['mlp']
        if use_ensemble is True:
            tf_models_list.append('ensemble')
        tf_models_dict = {tf_model: models_dict[tf_model] for tf_model in tf_models_list}
        for tf_model in tf_models_list:
            del mp_models_dict[tf_model]

    with mp.Manager() as manager:
        print('getting magecs via multiprocessing...')
        baseline_runs = plutils.generate_perturbation_predictions(
            mp_models_dict, x_validation_p, y_validation_p, baselines, features, mp_manager=manager)
        print('Done multiprocessing')

    if has_tf_models:
        print('getting magecs for TF models ...')
        tf_baseline_runs = plutils.generate_perturbation_predictions(
            tf_models_dict, x_validation_p, y_validation_p, baselines, features, mp_manager=None)

        baseline_runs = plutils.combine_baseline_runs(baseline_runs, tf_baseline_runs, baselines)

    baseline_to_scores_df, all_joined_dfs = plutils.score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, policy)

    df_logits_out = plutils.visualize_output(baseline_to_scores_df, baselines, features, out_type='logits')
    df_probs_out = plutils.visualize_output(baseline_to_scores_df, baselines, features, out_type='probs')

    # output_logits = {}
    # output_probs = {}

    # for baseline in baselines:
    #     if baseline is None:
    #         baseline = 0
    #     df_logits = pd.DataFrame.from_records(baseline_to_scores_df[baseline]['logits'])
    #     df_probs = pd.DataFrame.from_records(baseline_to_scores_df[baseline]['probs'])

    #     if baseline in [None, 0]:
    #         baseline = 1.0
    #     base_logits_strings = plutils.get_string_repr(df_logits, features)
    #     base_probs_strings = plutils.get_string_repr(df_probs, features)

    #     output_logits[baseline] = base_logits_strings
    #     output_probs[baseline] = base_probs_strings
    
    # # TODO: fix baselines upstream  to handle None as 0
    # if None in baselines:
    #     idx = baselines.index(None)
    #     baselines[idx] = 1.0

    # df_logits_out = plutils.produce_output_df(output_logits, features, baselines)
    # df_probs_out = plutils.produce_output_df(output_probs, features, baselines)

    # print(df_logits_out.head())
    # print(df_probs_out.head())

    return (df_logits_out, df_probs_out), all_joined_dfs
