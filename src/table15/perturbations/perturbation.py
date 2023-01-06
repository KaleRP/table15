from abc import ABC, abstractmethod

class Perturbation(ABC):
    @abstractmethod
    def run_perturbation():
        pass
    
    @abstractmethod
    def perturb_categorical():
        pass
    
    @abstractmethod
    def perturb_binary():
        pass
    
    @abstractmethod
    def perturb_numerical():
        pass
    
    @abstractmethod
    def score_comparison():
        pass