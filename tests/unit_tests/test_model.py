#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:32:12 2018

@author: jacaseyclyde
"""
import numpy as np

from mcorbit import model


class TestModelPointProbability(object):
    """
    Test the point to point probability model.
    """
    def setup_method(self):
        self.data = np.random.rand(10, 3)
        self.model = model.Model(self.data, np.random.rand(5))

    def test_inf_dist(self):
        assert all(self.model.model_pt_prob(np.array(3 * [np.inf])) == 0.)

    def test_neg_inf_dist(self):
        assert all(self.model.model_pt_prob(np.array(3 * [-np.inf])) == 0.)

    def test_zero_dist(self):
        idx = np.random.randint(len(self.data))
        probs = self.model.model_pt_prob(self.data[idx])
        assert probs[idx] == np.max(probs)


class TestModelProbability(object):
    """
    Test the point to orbit model probability model.
    """

    def setup_method(self):
        """
        Set up the :obj:`Model` object for testing with a random fake dataset.
        """
        # The initialization data for this set of tests doesn't matter at all
        self.data = np.random.rand(10, 3)
        self.model = model.Model(self.data, np.random.rand(5))

    def test_inf_dist(self):
        """
        Tests the posterior probability for a datapoint at infinity in all axes
        """
        model = np.array(10 * [np.array(3 * [np.inf])])
        prob = self.model.model_prob(model)
        assert all(prob == 0.)

    def test_neg_inf_dist(self):
        """
        Tests the posterior probability for a datapoint at -inf in all axes
        """
        model = np.array(10 * [np.array(3 * [-np.inf])])
        prob = self.model.model_prob(model)
        assert all(prob == 0.)


class TestLnLike(object):
    """
    Test the natural log of the likelihood function
    """
    def setup_method(self):
        """
        Set up the :obj:`Model` object for testing with a random fake dataset.
        """
        # The initialization data for this set of tests doesn't matter at all
        self.data = np.random.rand(10, 3)
        self.model = model.Model(self.data, np.random.rand(5))

    def test_inf_dist(self):
        """
        Tests the posterior probability for a datapoint at infinity in all axes
        """
        model = np.array(10 * [np.array(3 * [np.inf])])
        assert self.model.ln_like(model) == -np.inf

    def test_neg_inf_dist(self):
        """
        Tests the posterior probability for a datapoint at -inf in all axes
        """
        model = np.array(10 * [np.array(3 * [-np.inf])])
        assert self.model.ln_like(model) == -np.inf

    def test_no_dist(self):
        """
        Tests the posterior probability for a datapoint on the model
        """
        data_prob = self.model.ln_like(self.data)
        perturb_prob = self.model.ln_like(self.data + np.random.normal(0))
        assert data_prob >= perturb_prob


class TestLnPrior(object):
    """
    Test the natural log of the prior distribution
    """

    def setup_method(self):
        # The initialization data for this set of tests doesn't matter at all
        pspace = np.array([[-1., 1.],
                           [-1., 1.],
                           [-1., 1.],
                           [-1., 1.],
                           [-1., 1.]])

        self.model = model.Model(np.random.rand(6, 3), pspace)

    def test_bound(self):
        params = np.random.rand(5)
        ln_prior = self.model.ln_prior(params)

        assert ln_prior == np.log(2**(-5))

    def test_unbound(self):
        params = np.random.rand(5) + 1
        ln_prior = self.model.ln_prior(params)

        assert ln_prior == -np.inf
