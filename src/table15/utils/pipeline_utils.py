import multiprocessing as mp
from collections import defaultdict
from nis import cat
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import magec_utils as mg


def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        parsed_yaml = yaml.safe_load(file)
    return parsed_yaml

    
def get_from_configs(configs: Dict, key: str, param_type: str=None):
    key = key.upper()
    if param_type in configs and key in configs[param_type]:
        return configs[param_type][key]
    if key in configs['CONFIGS']:
        return configs['CONFIGS'][key]
    print(f'Warning: could not locate param {key} in configs')
    return None


def generate_data(configs: Dict):
    def impute(df):
        out = df.copy()
        cols = df.columns
        out[cols] = out[cols].replace(0, np.NaN)
        out[cols] = out[cols].fillna(out[cols].mean())
        return out

    csv_path = get_from_configs(configs, 'CSV_PATH')

    numerical_features = get_from_configs(configs, 'NUMERICAL', param_type='FEATURES')
    categorical_features = get_from_configs(configs, 'CATEGORICAL', param_type='FEATURES')
    binary_features = get_from_configs(configs, 'BINARY', param_type='FEATURES')
    target_feature = get_from_configs(configs, 'TARGET', param_type='FEATURES')

    random_seed = get_from_configs(configs, 'RANDOM_SEED', param_type='HYPERPARAMS')
    test_size = get_from_configs(configs, 'TEST_SIZE', param_type='HYPERPARAMS')

    df = pd.read_csv(csv_path)

    if random_seed is not None:
        np.random.seed(random_seed)

    non_numerical_features = binary_features # + categorical_features
    features = numerical_features + non_numerical_features

    x = df.loc[:, numerical_features]
    x = impute(x)
    x = pd.concat([x, df[non_numerical_features]], axis=1)

    Y = df.loc[:, target_feature]

    x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=test_size, random_state=random_seed)

    stsc = StandardScaler()

    xst_train = stsc.fit_transform(x_train[numerical_features])
    xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=numerical_features)
    xst_train = pd.concat([xst_train, x_train[non_numerical_features]], axis=1)
        
    xst_validation = stsc.transform(x_validation[numerical_features])
    xst_validation = pd.DataFrame(xst_validation, index=x_validation.index, columns=numerical_features)
    xst_validation = pd.concat([xst_validation, x_validation[non_numerical_features]], axis=1)

    # Format
    x_validation_p = xst_validation.copy()
    x_validation_p['timepoint'] = 0
    x_validation_p['case'] = np.arange(len(x_validation_p))
    x_validation_p.set_index(['case', 'timepoint'], inplace=True)
    x_validation_p = x_validation_p.sort_index(axis=1)

    y_validation_p = pd.DataFrame(Y_validation.copy())
    y_validation_p['timepoint'] = 0
    y_validation_p['case'] = np.arange(len(x_validation_p))
    y_validation_p.set_index(['case', 'timepoint'], inplace=True)
    y_validation_p = y_validation_p.sort_index(axis=1)

    # Format
    x_train_p = xst_train.copy()
    x_train_p['timepoint'] = 0
    x_train_p['case'] = np.arange(len(x_train_p))
    x_train_p.set_index(['case', 'timepoint'], inplace=True)
    x_train_p = x_train_p.sort_index(axis=1)

    y_train_p = pd.DataFrame(Y_train.copy())
    y_train_p['timepoint'] = 0
    y_train_p['case'] = np.arange(len(y_train_p))
    y_train_p.set_index(['case', 'timepoint'], inplace=True)
    y_train_p = y_train_p.sort_index(axis=1)

    return df, features, x_train_p, x_validation_p, y_train_p, y_validation_p


def create_mlp(x_train_p=None):
    mlp = Sequential()
    mlp.add(Dense(60, input_dim=len(x_train_p.columns), activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(30, input_dim=60, activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(1, activation='sigmoid'))
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mlp


def train_models(x_train_p, y_train_p, models, use_ensemble=False):
    """
    3 ML models for scaled data
    :param x_train_p:
    :param y_train_p:
    :return:
    """

    estimators = list()

    if 'lr' in models:
        lr = LogisticRegression(C=1.)
        lr.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('lr', lr))

    if 'rf' in models:
        rf = RandomForestClassifier(n_estimators=1000)
        rf.fit(x_train_p, y_train_p.values.ravel())
        sigmoidRF = CalibratedClassifierCV(RandomForestClassifier(n_estimators=1000), cv=5, method='sigmoid')
        sigmoidRF.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('rf', sigmoidRF))

    if 'mlp' in models:
        params = {'x_train_p': x_train_p}
        mlp = KerasClassifier(build_fn=create_mlp, x_train_p=x_train_p, epochs=100, batch_size=64, verbose=0)
        mlp._estimator_type = "classifier"
        mlp.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('mlp', mlp))
    
    if use_ensemble:
        # create our voting classifier, inputting our models
        ensemble = VotingClassifier(estimators, voting='soft')
        ensemble._estimator_type = "classifier"
        ensemble.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('ensemble', ensemble))
    
    models_dict = dict()
    for model_name, clf in estimators:
        models_dict[model_name] = clf
    
    return models_dict


def generate_perturbation_predictions(models_dict, x_validation_p, y_validation_p, baselines, features, mp_manager=None):
    is_multi_process = False
    run_dfs = dict()
    if mp_manager is not None:
        is_multi_process = True
        run_dfs = mp_manager.dict()
        processes = []
    
    keys = []
    for baseline in baselines:
        for model in models_dict.keys():
            key = model + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model + '_0'
            keys.append(key)
            clf = models_dict[model]
            if is_multi_process is False and model in ['mlp', 'lstm', 'ensemble']:
                    if model in ['lstm']:
                        clf = clf.model
                    run_dfs[key] = run_magecs_single(clf, x_validation_p, y_validation_p, model, key, baseline, features)
            elif is_multi_process is True:
                p = mp.Process(name=key, target=run_magecs_multip, 
                    args=(run_dfs, clf, x_validation_p, y_validation_p, model, baseline, features))
                processes.append(p)
            else:
                raise ValueError(f'Cannot run {key} through multiprocessing')
        
    if is_multi_process:
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    baseline_runs = defaultdict(list)
    for key in keys:
        baseline = key.split('_')[1]
        if baseline[0] == 'p':
            baseline = int(baseline[1:]) / 100
        else:
            baseline = int(baseline)
        configs_check = baseline
        if baseline == 0:
            configs_check = None
        assert configs_check in baselines
        baseline_runs[baseline].append(run_dfs[key])
    
    return baseline_runs


def run_magecs_single(clf, x_validation_p, y_validation_p, model_name, key, baseline=None, features=None):
    print('Starting single:', key)
    if model_name == 'lstm':
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline, timeseries=True)
    else:
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline)
    print('Magecs for {} computed...'.format(key))
    magecs = mg.normalize_magecs(magecs, features=features, model_name=model_name)
    print('Magecs for {} normalized...'.format(key))
    magecs = magecs.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', key)
    return magecs
    

def run_magecs_multip(return_dict, clf, x_validation_p, y_validation_p, model_name, baseline=None, features=None):
    p_name = mp.current_process().name
    print('Starting multi:', p_name)
    if model_name == 'lstm':
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline, timeseries=True)
    else:
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline)
    print('Magecs for {} computed...'.format(p_name))
    magecs = mg.normalize_magecs(magecs, features=features, model_name=model_name)
    print('Magecs for {} normalized...'.format(p_name))
    magecs = magecs.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', p_name)
    return_dict[p_name] = magecs


def combine_baseline_runs(main_dict, to_combine_dict, baselines):
    for baseline in baselines:
        main_dict[baseline].extend(to_combine_dict[baseline])
    return main_dict


def score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, policy):
    baseline_to_scores_df = {}
    all_joined_dfs = {}
    for baseline, model_runs in baseline_runs.items():
        baseline_joined = mg.magec_models(*model_runs,
                            Xdata=x_validation_p,
                            Ydata=y_validation_p,
                            features=features)
        baseline_ranked_df = mg.magec_rank(baseline_joined, rank=len(features), features=features, models=models)
        scores_df = agg_scores(baseline_ranked_df, policy=policy, models=models)

        all_joined_dfs[baseline] = baseline_joined
        baseline_to_scores_df[baseline] = scores_df
    return baseline_to_scores_df, all_joined_dfs


def agg_scores(ranked_df, policy='mean', models=('mlp', 'rf', 'lr')):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, use_weights=False, policy=policy)
        out.append(scores)
    
    return pd.DataFrame.from_records(out)


def get_string_repr(df, feats):
    base_strings = []
    for feat in feats:
        mean = round(df[feat].mean(), 4)
        # std = round(df[feat].std(), 4)
        sem = round(df[feat].sem(), 4)
        # string_repr = f'{mean} +/- {std}'
        string_repr = f'{mean} ({sem})'
        base_strings.append(string_repr)
    return base_strings


def produce_output_df(output, features, baselines):
    df_out = pd.DataFrame.from_records(output)
    df_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    df_out = df_out.rename(columns={'0': 'full'})
    df_out = df_out[cols]
    return df_out

def visualize_output(baseline_to_scores_df, baselines, features,  out_type='logits'):
    output = {}
    for baseline in baselines:
        df_out = pd.DataFrame.from_records(baseline_to_scores_df[baseline][out_type])
        output[baseline] = get_string_repr(df_out, features)
    
    # TODO: fix baselines upstream  to handle None as 0
    formatted_baselines = baselines.copy()

    df_out =  produce_output_df(output, features, formatted_baselines)
    return df_out
