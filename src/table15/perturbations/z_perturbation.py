from src.table15.perturbations.perturbation import Perturbation
import numpy as np
import pandas as pd

class Z_Perturbation(Perturbation):
    def __init__(self) -> None:
        super().__init__()
        
    def score_comparison(self, x_orig, x_perturb):
        return x_perturb - x_orig
    
    
    def z_perturbation(self, model, target_data, features, feature_type, set_feature_values, baseline=1.0):
        '''
        Main method for computing a MAgEC. Assumes 'scaled/normalized' features in target data.
            Supporting 2 types of variables:
            - numeric / floats
            - binary / boolean
            Default score_comparison subtracts perturbed output from original.
            For a binary classification task, where 1 denotes a "bad" outcome, a good perturbation
            is expected to result in a negative score_comparison (assuming monotonic score_preprocessing).
        :param model:
        :param target_data:
        :param score_comparison:
        :param sort_categories:
        :param categories:
        :param features:
        :param epsilon_value:
        :param baseline: whether to compute baseline MAgECS, None as default, 0.01 for 1% perturbation
        :return:
        '''
        self.perform_data_checks(target_data, feature_type, features)

        prob_deltas_per_cell = pd.DataFrame(index=target_data.index, columns=target_data.columns)
        
        # Predict for original data
        base_df = target_data.copy()
        base_df = self.model_predict_probs_and_logits(base_df, model, label="orig")
        
        for var_name in features:
            # Predict for perturbed feature data
            perturb_df = target_data.copy()
            perturb_df = self.run_perturbation(
                perturb_df, var_name, baseline, set_feature_values, feature_type)
            perturb_df = self.model_predict_probs_and_logits(perturb_df, model, label="perturb")
            
            # logits
            logit_orig = base_df['logit_orig']
            logit_perturb = perturb_df['logit_perturb']
            logit_diff = self.score_comparison(logit_orig, logit_perturb)
            # store
            prob_deltas_per_cell[var_name] = logit_diff
            prob_deltas_per_cell[f'perturb_{var_name}_prob'] = perturb_df['probs_perturb']
            prob_deltas_per_cell['orig_prob'] = base_df['probs_orig']

        return prob_deltas_per_cell.astype(float)
    
    
    def perform_data_checks(self, target_data, feature_type, features):
        assert 'case' in target_data.index.names, "missing 'case' from index"
        
        features = np.asarray(features)
        assert len(features) > 0, f"No features here to perturb. Feature type: {feature_type}."
        

        # assert features only have 2 values
        binary = target_data[features].apply(lambda x: len(np.unique(x)), ) <= 2
        binary = binary[binary].index.tolist()
        assert (feature_type in ["binary", "categorical"] and len(binary) > 0) or (feature_type not in ["binary", "categorical"] and len(binary) == 0), (
            f"Mismatch between binary feature_type = {feature_type} and len(binary) = {len(binary)}")

        epsilons = dict()
        for var_name in features:
            if var_name in binary:
                epsilon = target_data[var_name].unique().tolist()
                epsilons[var_name] = epsilon
                if type(epsilons[var_name]) is list and len(epsilons[var_name]) <= 2:
                    assert feature_type in ["binary", "categorical"]
                if "__cat__" in var_name:
                    assert feature_type == "categorical"
            else:    
                assert var_name not in epsilons
                assert feature_type == "numerical"
    
    def run_perturbation(self, df, var_name, baseline, set_feature_values, feature_type):
        # perturb to baseline conditions
        if feature_type == "numerical": 
            return self.perturb_numerical(df, var_name, baseline, set_feature_values)
        elif feature_type == "binary": 
            return self.perturb_binary(df, var_name, baseline)
        elif feature_type == "categorical": 
            return self.perturb_categorical(df, var_name, baseline)
        raise
    
    
    def perturb_categorical(self, df, var_name, baseline, delimiter="__cat__"):
        assert baseline == 1.0, "Baseline always 1.0 for categorical features"
        perturbed = df.copy()
        perturbed[var_name] = baseline
        # Get other similar categories to reassign 0 value
        cat_name = var_name.split(delimiter)[0]
        similar_cats = [col for col in perturbed.columns if delimiter in col and col.split(delimiter)[0] == cat_name]
        for sim_cat in similar_cats:
            if sim_cat != var_name:
                perturbed[sim_cat] = 0.0
        return perturbed


    def perturb_binary(self, df, var_name, baseline):
        assert baseline in [0.0, 1.0], "Baseline either 1.0 or 0.0 for binary features"
        perturbed = df.copy()
        perturbed[var_name] = baseline
        return perturbed


    def perturb_numerical(self, df, var_name, baseline, set_feature_values=None):
        perturbed = df.copy()
        if set_feature_values and var_name in set_feature_values:
            set_val = set_feature_values[var_name]
        else:
            set_val = 0.0
        curr_val = perturbed.loc[:, var_name]
        pert_dist = curr_val - set_val
        new_val = self.perturb_num_series_with_baseline_scaling(curr_val, pert_dist, baseline)
        perturbed[var_name] = new_val
        return perturbed


    def perturb_num_series_with_baseline_scaling(self, curr_val, perturbation_distance, baseline_scaling):
        return curr_val - (perturbation_distance * float(baseline_scaling))
    
    
    def model_predict_probs_and_logits(self, df, model, label=None):
        probs = self.predict(model, df)
        logits = self.get_logit_ln(probs)
        df_cols = df.columns
        df['probs_{}'.format(label)] = probs
        df['logit_{}'.format(label)] = logits
        df = df.drop(df_cols, axis=1)
        return df
    
    def predict(self, model, data):
        """
        Model output (predicted) probabilities.
        Wrapper for predict_proba function in scikit-learn models.
        When a model does not have a predict_proba use predict interface.
        """
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(data)
            if len(probs.shape) == 2 and probs.shape[1] == 2:
                probs = probs[:, 1].ravel()
            else:
                probs = probs.ravel()
        else:
            probs = np.array(model.predict(data))
        return probs
    
    def get_logit_base2(self, prob, eps=1e-16):
        return np.log2((prob+eps)/(1-prob+eps))


    def get_logit_ln(self, prob, eps=1e-16):
        return np.log((prob+eps)/(1-prob+eps))
