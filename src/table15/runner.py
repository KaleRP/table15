import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import List

import pandas as pd
from pathlib import Path

import table15.utils.pipeline_utils as plutils
from table15.configs import PipelineConfigs
from table15.utils.data_tables import DataTables
from table15.utils.models_container import ModelsContainer


def run(pipeline_configs_path: str='./configs/pipeline_configs/pima.yaml', output_path='./output') -> List[pd.DataFrame]: 
    """ Application's main driver method. Steps include:
    1) Setup configurations and pipeline parameters
    2) Generate input data and supporting metrics
    3) Build, train and store feature importance of each model
    4) Generate table by performing feature perturbations for each feature type
    5) Display and return table

    Args:
        pipeline_configs_path (str, optional): _description_. Defaults to './configs/pima_diabetes.yaml'.

    Returns: List[pd.DataFrame]: List of pandas dataframes representing output tables
    """
    pipeline_configs = PipelineConfigs(pipeline_configs_path)
    
    data_configs_path = pipeline_configs.get_from_configs("DATA_CONFIGS_PATH", param_type="DATA")
    models_configs_paths = pipeline_configs.get_from_configs('MODEL_CONFIGS_PATHS', param_type='MODELS')
    use_multiprocessing = pipeline_configs.get_from_configs('USE_MULTIPROCESSING', param_type='DEBUGGING',
                                                            default=True)
    
    use_feature_importance_scaling = pipeline_configs.get_from_configs('USE_FEATURE_IMPORTANCE_SCALING', 
                                                                       param_type='MODELS')
    ensemble_configs_path = pipeline_configs.get_from_configs('ENSEMBLE_CONFIGS_PATH', param_type='MODELS')
    
    data_tables = DataTables() \
        .set_data_configs(data_configs_path) \
        .generate_data()

    # Generate and train models
    models_container = ModelsContainer() \
        .populate_data_tables(data_tables.x_train, data_tables.Y_train, data_tables.x_test) \
        .load_models(models_configs_paths, ensemble_configs_path=ensemble_configs_path) \
        .train_models() \
        .store_feature_importance_from_models(use_feature_importance_scaling=use_feature_importance_scaling)

    df_out_by_feature_types = []
    feature_types = ["numerical", "binary", "categorical", "grouped"]
    for feature_type in feature_types:
        df_out = plutils.generate_table_by_feature_type(data_tables, models_container, 
                                                                               feature_type=feature_type, 
                                                                               use_multiprocessing=use_multiprocessing)
        df_out_by_feature_types.append(df_out)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df_out.to_csv(Path(f'{output_path}/{feature_type}.csv').absolute())

    for df in df_out_by_feature_types:
        if df is not None:
            print(df.head(20))
        
    print("Done!")
    return df_out_by_feature_types


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        pipeline_configs_path = args[0]
        output_path = args[1] if len(args) > 1 else './output'
    else:
        pipeline_configs_path = 'src/table15/configs/pipeline_configs/pima.yaml'
        output_path = './output'
    df_logits_out, all_joined_dfs = run(pipeline_configs_path=pipeline_configs_path, output_path=output_path)
