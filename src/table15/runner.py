import multiprocessing as mp
import os
import sys
import warnings
from multiprocessing import set_start_method

import utils.pipeline_utils as plutils


def run(configs_path='./configs/pima_diabetes.yaml'):
    if not os.path.isabs(configs_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        configs_path = os.path.join(script_dir, configs_path)

    warnings.filterwarnings('ignore')
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    
    # TODO: adjust spawn method to start WITH multiprocessing. Most likely with mp.Pool()

    print('This is Version: 0.0.22')

    configs = plutils.yaml_parser(configs_path)
    models = plutils.get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    use_ensemble = plutils.get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')
    
    df, features, x_train_p, x_validation_p, y_train_p, y_validation_p = plutils.generate_data(configs)

    # Train models
    print('Training models ...')
    models_dict = plutils.train_models(x_train_p, y_train_p, models, use_ensemble=use_ensemble)
    print(f'Finished training models {list(models_dict.keys())}')

    df_logits_out_num, all_joined_dfs_num = plutils.generate_table_by_feature_type(
        configs, x_validation_p, y_validation_p, models_dict, feature_type='numerical')
    
    df_logits_out_bin, all_joined_dfs_bin = plutils.generate_table_by_feature_type(
        configs, x_validation_p, y_validation_p, models_dict, feature_type='binary')

    return [df_logits_out_num, df_logits_out_bin], [all_joined_dfs_num, all_joined_dfs_bin]


if __name__ == '__main__':
    # config_path = sys.argv[1]
    config_path = '/Users/ag46548/tmp/t15_configs/t15_stroke.yaml'
    if config_path:
        df_logits_out, all_joined_dfs = run(configs_path=config_path)
    else:
        df_logits_out, all_joined_dfs = run()

    print('Done!')
