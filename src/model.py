#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:08:27 2018

@author: jacaseyclyde
"""
import numpy as np

np.set_printoptions(precision=5,threshold=np.inf)

global datafile
global outpath
global stamp

datafile='../dat/CLF-Sim.csv'
outpath='../out/'

def PointPointProb(d,e):
    '''
    Returns the probability of point d being generated by a multivariate
    gaussian distribution centered on point e, with a covariance matrix cov.
    See the definition of multivariate gaussian distributions in
    Statistics, Data Mining, and Machine Learning in Astronomy, Ivezic et al. 2014
    '''
    x = d - e
    
    exp = np.exp(-0.5 * np.matmul(x, np.matmul(H,x)))
    
    return exp
            
def PointModelProb(d,E):
    '''
    Returns the marginalized probability of a point d being generated by a 
    given model E
    '''
    
    pdE = 0.    # probability of getting point d, given model E
    
    for e in E:
        pdE += PointPointProb(d,e)
    
    return pdE
    
def lnLike(theta):
    '''
    Returns the log-likelihood that the dataset D was generated by the model
    definted by parameters theta. i.e.
    lnL = ln(p(D|theta))
    D should be formatted as [[x,y]]
    '''
    E = OrbitFunc(theta)
    
    lnlike = 0.
    
    for d in data:
        lnlike += np.log(PointModelProb(d,E))
    
    return lnlike

def lnPrior(theta):
    '''
    The log-likelihood of the prior. Currently assuming uniform
    '''
    #aop,loan, inc, a, e = theta
    if ((theta >= pos_min).all() and (theta < pos_max).all()):
        return 0.0
    return -np.inf

def lnProb(theta):
    '''
    Returns the total log probability that dataset D could have been generated
    by the elliptical model with parameters theta
    '''
    lp = lnPrior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnLike(theta)
    return lp + ll