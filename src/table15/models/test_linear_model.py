from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    
    @abstractmethod
    def predict_proba(self, data):
        pass

class TestBasicModel(Model):
    def predict_proba(self, data):
        return np.array([[0.5] * 2] * len(data))
        