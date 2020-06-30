from abc import ABC, abstractmethod 
import numpy as np 

class DFLDynamicPlant(ABC):
    def __init__(self):
        pass

    # @property
    # @abstractmethod
    # def A_x(self):
    #     pass

    # @property
    # @abstractmethod
    # def B_x(self):
    #     pass

    @abstractmethod
    def f(self,x,u,t):
        '''
        This function defines the time transition of a continuous time dynamic system.
        x: state
        u: exogenous input
        t: time
        '''
        pass

    @abstractmethod
    def g(self,x,u,t):
        '''
        This function defines the observed output of the system.
        x: state
        u: exogenous input
        t: time
        '''
        pass

    @abstractmethod
    def phi(self,x,u,t):
        '''
        outputs the values of the auxiliary variables
        '''
        pass

    # @abstractmethod
    # def gamma(self,x,u,t):
    #     '''
    #     transition equation of the auxiliary variables
    #     '''
    #     pass





