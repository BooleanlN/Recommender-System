from abc import ABCMeta,abstractmethod
import numpy as np
class Model():
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def train(self,epochs):
        pass

    @abstractmethod
    def predict(self,X_test):
        pass
