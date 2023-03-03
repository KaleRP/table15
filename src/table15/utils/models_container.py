from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from keras.layers import Dense, Dropout
from keras.models import Sequential

from configs import ModelConfigs
from models.model import Model
from models.model_factory import ModelFactory


class ModelsContainer:
    def __init__(self):
        self.models: List[Model] = []
        self.models_dict: Dict[str, Model] = {}
        self.model_feat_imp_dict: Dict[str, Dict[str, float]] = defaultdict(dict)

    def load_models(self, model_configs_paths: List[str], ensemble_configs_path: Optional[str]=None) -> ModelsContainer:
        """Instantiate and load all models specified in Pipeline Configs. 
        Additionally, an VotingClassifier ensemble of all models can be optionally added.

        Args:
            model_configs_paths (List[str]): path to each Model Configs used in run, where each path is a seperate model
            ensemble_configs_path (Optional[str], optional): Path to ensemble model configs. This Voting Classifier model 
            comprises of all previously used models. Defaults to None.

        Returns:
            ModelsContainer: self
        """
        for config_path in model_configs_paths:
            m_config = ModelConfigs(config_path)
            model = self.construct_model(m_config) \
                .instantiate_model()
            self.models.append(model)
        if ensemble_configs_path is not None:
            m_config = ModelConfigs(ensemble_configs_path)
            estimators = [(m.name, m.model) for m in self.models]
            model = self.construct_model(m_config) \
                .set_estimators(estimators) # type: ignore
            model = model.instantiate_ensemble_model() 
            self.models.append(model)
        return self

    def populate_data_tables(self, x_train: pd.DataFrame, Y_train: pd.DataFrame, x_test: pd.DataFrame=None) -> ModelsContainer:
        """Setter function to populate data membership.

        Args:
            x_train (pd.DataFrame): Pandas dataframe with input columns to model
            Y_train (pd.DataFrame): Pandas dataframe with output columns to model
            x_test (pd.DataFrame, optional): Pandas dataframe for measuring perturbations. Defaults to None.

        Returns:
            ModelsContainer: self
        """
        self.x_train = x_train
        self.Y_train = Y_train
        self.x_test = x_test
        return self

    def construct_model(self, model_configs: ModelConfigs) -> Model:
        """Wrapper method to call ModelFactory and construct a model based on Model Configs

        Args:
            model_configs (ModelConfigs): Configs object specific to model building

        Returns:
            Model: Constructed model
        """
        return ModelFactory.construct_model(model_configs)
    
    def train_models(self) -> ModelsContainer:
        """Trains each model in `self.models` using training set and stores each model

        Raises:
            ValueError: Raises error if no models exist to train

        Returns:
            ModelsContainer: self
        """
        if len(self.models) == 0:
            raise ValueError("No models generated to train...")
        print('Training models ...')
        for model in self.models:
            model = model.fit(self.x_train, self.Y_train)
            self.models_dict[model.name] = model
            print(f"Finished training {model.name} model")
        print(f'Finished generating models {list(self.models_dict.keys())}')
        return self
    
    def store_feature_importance_from_models(self, use_feature_importance_scaling: bool=True) -> ModelsContainer:
        """Extracts feature importances from models through model-specific implementations. Stores these
            into `self.model_feat_imp_dict`

        Args:
            use_feature_importance_scaling (bool, optional): Only performs feature importance scaling extraction
                if this is set to True in Pipeline Configs. Defaults to True.

        Raises:
            ValueError: Raises error if no models exist

        Returns:
            ModelsContainer: self
        """
        if len(self.models_dict) == 0:
            raise ValueError("Models have not been trained yet")
        if use_feature_importance_scaling is True:
            features = self.x_train.columns
            for model_name, model in self.models_dict.items():
                model_feature_importances = model.extract_feature_importances()
                model_feature_importances = self.l2_normalize_list(model_feature_importances, is_abs=True)
                feature_to_importance = dict(zip(features, model_feature_importances))
                self.model_feat_imp_dict[model_name] = feature_to_importance 
            print("Extracted model feature importances")
        return self
    
    def l2_normalize_list(self, li: np.array, is_abs: bool=False) -> np.array:
        """Calculates L2 (Euclidean) from a list of numbers. Used during feature importance scaling.

        Args:
            li (np.array): Numpy array
            is_abs (bool, optional): Flag to only output absolute values of l2 normalization. Defaults to False.

        Returns:
            np.array: Sample numpy array as input, but with each element normalized
        """
        norm = np.linalg.norm(np.array(li))
        return np.abs(li / norm) if is_abs is True else li / norm
    

    def get_shap_values(self, model: Model, explainer_type: str) -> np.array:
        """NB: Currently unused. Can be reimplimented to calculate SHAP values of various model types.

        Args:
            model (Model): An implemented Model object.
            explainer_type (str): Describes what explainer method to use on model.

        Returns:
            np.array: Normalized shap value means.
        """
        if explainer_type == 'kernel':
            def model_predict(data_asarray):
                data_asframe = pd.DataFrame(data_asarray, columns=model_features)
                return model.predict(data_asframe)
            NUM_CENTROIDS = 10
            model_features = self.x_train.columns
            x_train_kmeans = shap.kmeans(self.x_train, NUM_CENTROIDS)
            explainer = shap.KernelExplainer(model_predict, np.array(x_train_kmeans.data))
        elif explainer_type == 'deep':
            background = shap.sample(self.x_train.to_numpy(), 50)
            explainer = shap.DeepExplainer(model, background)
            
        shap_values = explainer.shap_values(shap.sample(self.x_test.to_numpy(), 50))
        shap_means = np.mean(shap_values, axis=0)
        l2_norm = np.linalg.norm(shap_means)
        normalized_shap_means = np.abs(shap_means) / l2_norm
        return normalized_shap_means.ravel()
