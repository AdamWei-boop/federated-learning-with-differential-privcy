# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 20:04:02 2021

@author: adamwei


"""
import sys
sys.dont_write_bytecode = True
import os, sys, time, traceback, math
import numpy as np

from opacus import privacy_analysis
# https://github.com/pytorch/opacus
# https://github.com/tensorflow/privacy

#from pathos.multiprocessing import ProcessingPool
# https://pathos.readthedocs.io/en/latest/pathos.html
from functools import partial

# For a given noise_multiplier, compute epsilon
# Or, for a given epsilon, search for noise_multiplier
# In both cases, we need to search for the best alpha
# Parallel computing

class ComputeNoiseEpsA(object):
    def __init__(self, num_workers=1):
        super(ComputeNoiseEpsA, self).__init__()
        self.num_processes = num_workers

        self.q = 0.1
        self.num_steps = 128
        self.alpha_list = list()
        self.rdp_list = list()
        self.delta = 1e-5

        self.best_epsilon = 0.0
        self.best_alpha = 0.0

    # For a given noise_multiplier, compute epsilon
    def compute_steps(self, noise_multiplier=1.0, alpha_list = [2], basic_steps= 5, epsilon = 3):
        
        num_steps_list = list( np.arange(basic_steps, basic_steps*200, basic_steps) )
        for num_steps in num_steps_list:
            self.rdp_list = privacy_analysis.compute_rdp(q=self.q, noise_multiplier=noise_multiplier,
                             steps = num_steps, orders=self.alpha_list) 
            temp_epsilon, temp_alpha = privacy_analysis.get_privacy_spent(orders=self.alpha_list, rdp=self.rdp_list, delta=self.delta)            
            
            # print('Eps:', self.alpha_list, temp_epsilon, temp_alpha)
            
            if temp_epsilon > epsilon:               
                num_steps = num_steps-basic_steps
                break
            elif temp_epsilon == epsilon:
                break
        
        return math.floor(num_steps/basic_steps)
        
    def compute_epsilon_alpha(self, noise_multiplier=1.0, num_workers=1):
        temp_alpha, temp_epsilon = None, None
        # if num_processes <= 1:
        """
        def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
            ) -> Union[List[float], float]:
        """
        self.rdp_list = privacy_analysis.compute_rdp(q=self.q, noise_multiplier=noise_multiplier,
                                steps=self.num_steps, orders=self.alpha_list)

        """
        def get_privacy_spent(
        orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
        ) -> Tuple[float, float]:
        """
        # else:
        #     process_pool = ProcessingPool(num_processes)
        #     """
        #     def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
        #         ) -> Union[List[float], float]:
        #     """
        #     # Multi-processing
        #     def local_rdp(orders, q, noise_multiplier, steps):
        #         return privacy_analysis.compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
            
        #     part_func_rdp = partial( local_rdp, q=self.q, noise_multiplier=noise_multiplier, steps=self.num_steps )
        #     self.rdp_list = list( process_pool.map(part_func_rdp, self.alpha_list) )
        #     """
        #     def get_privacy_spent(
        #     orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
        #     ) -> Tuple[float, float]:
           
        #     def local_get_epsilon(rdp, orders, delta):
        #         return privacy_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        #     part_func_eps = partial( local_get_epsilon, orders=self.alpha_list, delta=self.delta )
        #     tuple_list = list(process_pool.map(part_func_eps, self.list_rdp_list))
        #      """
        temp_epsilon, temp_alpha = privacy_analysis.get_privacy_spent(orders=self.alpha_list, rdp=self.rdp_list, delta=self.delta)
        temp_rdp = self.rdp_list[list(self.alpha_list).index(temp_alpha)]
        return temp_epsilon, temp_alpha, temp_rdp
    
    # For a given epsilon, search for the best noise_multiplier
    def compute_noise_mul(self, target_eps=1.0, search_list=[1.0, 2.0], num_workers=4):
        best_noise_mul, best_epsilon, best_alpha = [None]*3
        for noise_multiplier in search_list:
            temp_epsilon, temp_alpha, temp_rdp = self.compute_epsilon_alpha(noise_multiplier=noise_multiplier, num_workers=num_workers)
            if temp_epsilon < target_eps + 1e-2:
                best_noise_mul = noise_multiplier
                best_epsilon = temp_epsilon
                best_alpha = temp_alpha
                if best_noise_mul <= 0.1:
                    return best_noise_mul, best_epsilon, best_alpha, temp_rdp
                else:
                    # Try to look back a little bit
                    print("Refining noise_multiplier")
                    sub_list = list( np.arange((best_noise_mul-0.1), best_noise_mul, 0.001) )
                    for noise_multiplier in sub_list:
                        temp_epsilon, temp_alpha, temp_rdp = self.compute_epsilon_alpha(noise_multiplier=noise_multiplier, num_workers=num_workers)
                        if temp_epsilon < target_eps + 1e-2:
                            best_noise_mul = noise_multiplier
                            best_epsilon = temp_epsilon
                            best_alpha = temp_alpha
                            return best_noise_mul, best_epsilon, best_alpha, temp_rdp
                return best_noise_mul, best_epsilon, best_alpha, temp_rdp
        return [None]*4