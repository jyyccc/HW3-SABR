# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price=self.price(strike,spot,texp)
        return self.bsm_model.impvol(price,strike,spot,texp)
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1,nsamples=10000,nsteps=500,random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''

        time_interval=texp/nsteps
        time_coeff=np.sqrt(time_interval)
        rho_star=np.sqrt(1-self.rho**2)
        np.random.seed(random_seed)
        prices=np.zeros(nsamples)
        for i in range(nsamples):
            z=np.random.normal(size=nsteps)*time_coeff
            x=np.random.normal(size=nsteps)*time_coeff
            sigma_t=np.zeros(nsteps+1)
            s_t=np.zeros(nsteps+1)
            sigma_t[0]=self.sigma
            s_t[0]=spot
            for j in range(nsteps):
                sigma_t[j+1]=sigma_t[j]+self.vov*sigma_t[j]*z[j]
                s_t[j+1]=s_t[j]+sigma_t[j]*s_t[j]*(self.rho*z[j]+rho_star*x[j])
            prices[i]=s_t[nsteps]
        price_final=np.zeros(strike.size)
        
        for i in range(strike.size):
            price_final[i] = np.mean( np.fmax(cp*(prices - strike[i]), 0) )
        return price_final

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        price=self.price(strike,spot,texp)
        return self.normal_model.impvol(price,strike,spot,texp)
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1,nsamples=10000,nsteps=500,random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        time_interval=texp/nsteps
        time_coeff=np.sqrt(time_interval)
        rho_star=np.sqrt(1-self.rho**2)
        np.random.seed(random_seed)
        prices=np.zeros(nsamples)
        for i in range(nsamples):
            z=np.random.normal(size=nsteps)*time_coeff
            x=np.random.normal(size=nsteps)*time_coeff
            sigma_t=np.zeros(nsteps+1)
            s_t=np.zeros(nsteps+1)
            sigma_t[0]=self.sigma
            s_t[0]=spot
            for j in range(nsteps):
                sigma_t[j+1]=sigma_t[j]+self.vov*sigma_t[j]*z[j]
                s_t[j+1]=s_t[j]+sigma_t[j]*(self.rho*z[j]+rho_star*x[j])
            prices[i]=s_t[nsteps]
        price_final=np.zeros(strike.size)
        
        for i in range(strike.size):
            price_final[i] = np.mean( np.fmax(cp*(prices - strike[i]), 0) )
        return price_final

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price=self.price(strike,spot,texp)
        return self.bsm_model.impvol(price,strike,spot,texp)
    
    def price(self, strike, spot, texp=None, cp=1,nsamples=10000,nsteps=500,random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        time_interval=texp/nsteps
        time_coeff=np.sqrt(time_interval)
        rho_star=np.sqrt(1-self.rho**2)
        np.random.seed(random_seed)
        prices=np.zeros(nsamples)
        for i in range(nsamples):
            z=np.random.normal(size=nsteps)*time_coeff
            sigma_t=np.zeros(nsteps+1)
            sigma_t[0]=self.sigma
            for j in range(nsteps):
                sigma_t[j+1]=sigma_t[j]+self.vov*sigma_t[j]*z[j]
            v_t=np.square(sigma_t)
            V_T=0
            for j in range(nsteps):
                V_T+=(v_t[j]+v_t[j+1])/2*time_interval
            sigma_T=sigma_t[nsteps]
            X1=np.random.normal()
            prices[i]=spot*np.exp(self.rho/self.vov*(sigma_T-self.sigma)-0.5*V_T+rho_star*np.sqrt(V_T)*X1)
        price_final=np.zeros(strike.size)
        
        for i in range(strike.size):
            price_final[i] = np.mean( np.fmax(cp*(prices - strike[i]), 0) )
        return price_final

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price=self.price(strike,spot,texp)
        return self.normal_model.impvol(price,strike,spot,texp)
        
    def price(self, strike, spot, texp=None, cp=1,nsamples=10000,nsteps=500,random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        time_interval=texp/nsteps
        time_coeff=np.sqrt(time_interval)
        rho_star=np.sqrt(1-self.rho**2)
        np.random.seed(random_seed)
        prices=np.zeros(nsamples)
        for i in range(nsamples):
            z=np.random.normal(size=nsteps)*time_coeff
            sigma_t=np.zeros(nsteps+1)
            sigma_t[0]=self.sigma
            for j in range(nsteps):
                sigma_t[j+1]=sigma_t[j]+self.vov*sigma_t[j]*z[j]
            v_t=np.square(sigma_t)
            V_T=0
            for j in range(nsteps):
                V_T+=(v_t[j]+v_t[j+1])/2*time_interval
            sigma_T=sigma_t[nsteps]
            X1=np.random.normal()
            prices[i]=spot+self.rho/self.vov*(sigma_T-self.sigma)+rho_star*np.sqrt(V_T)*X1
        price_final=np.zeros(strike.size)
        
        for i in range(strike.size):
            price_final[i] = np.mean( np.fmax(cp*(prices - strike[i]), 0) )
        return price_final
