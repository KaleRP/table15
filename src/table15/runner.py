import pandas as pd
import multiprocessing
from multiprocessing import set_start_method
from . import magec_utils as mg
from . import pima_utils as pm
from . import pipeline_utils as plutils
import time
import warnings

def run(configs_path='../configs/pima_diabetes.yaml'):
    warnings.filterwarnings('ignore')
    set_start_method("spawn")

    print('This is Version: 0.0.3')

    configs = plutils.yaml_parser(configs_path)
    baselines = plutils.get_from_configs(configs, 'BASELINES')
    policy = plutils.get_from_configs(configs, 'POLICY')
    features = plutils.get_from_configs(configs, 'FEATURES')
    models = plutils.get_from_configs(configs, 'MODELS')

    pima, x_train, x_validation, stsc, x_train_p, x_validation_p, y_train_p, y_validation_p = pm.pima_data(configs)
    print(x_train_p.shape)
    print(y_train_p.shape)

    # Train models
    models_dict = pm.pima_models(x_train_p, y_train_p, models)

    print('getting magecs...')
    with multiprocessing.Manager() as manager:
        run_dfs = manager.dict()
        processes = []
        keys = []
        for model in models_dict.keys():
            for baseline in baselines:
                print(baseline)
                key = model + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model + '_0'
                keys.append(key)
                clf = models_dict[model]
                if model in ['mlp', 'lstm']:
                    clf = clf.model
                p = multiprocessing.Process(name=key, 
                                            target=run_magecs, 
                                            args=(run_dfs, clf, x_validation_p, y_validation_p, model, baseline))
                p.start()
                time.sleep(1)
                # processes.append(p)
        
        # for p in processes:
        #     p.start()
        for p in processes:
            p.join()

        joined = mg.magec_models(*run_dfs.values(),
                                Xdata=x_validation_p, 
                                Ydata=y_validation_p, 
                                features=features)

    all_ranked_df = mg.magec_rank(joined, rank=len(features), features=features)
    scores_df = agg_scores(all_ranked_df, policy=policy)
    return scores_df, joined


def agg_scores(ranked_df, policy='mean', models=('mlp', 'rf', 'lr')):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, use_weights=False, policy=policy)
        out.append(scores)
    
    return pd.DataFrame.from_records(out)

def run_magecs(return_dict, clf, x_validation_p, y_validation_p, model_name, baseline=None):
    p_name = multiprocessing.current_process().name
    print('Starting:', p_name)
    if model_name == 'lstm':
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline, timeseries=True)
    else:
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline)
    print('Magecs for {} computed...'.format(p_name))
    magecs = mg.normalize_magecs(magecs, features=None, model_name=model_name)
    print('Magecs for {} normalized...'.format(p_name))
    magecs = magecs.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', p_name)
    return_dict[p_name] = magecs
