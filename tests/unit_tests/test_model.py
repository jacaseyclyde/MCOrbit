#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:32:12 2018

@author: jacaseyclyde
"""
import numpy as np

from mcorbit import model


class TestPointPointProbability(object):
    """
    Test the point to point probability model.
    """

    def setup_method(self):
        """
        Set up the :obj:`Model` object for testing with a random fake dataset.
        """
        self.data = np.random.rand(10, 3)
        self.model = 10 *  np.zeros(5)
        self.model = model.Model(self.data, self.model)

    def test_no_dist(self):
        """
        Tests the probability for a datapoint on top of the model point
        """
        test_data_pt = np.zeros(3)
        test_model_pt = np.zeros(3)

        prob = self.model.point_point_prob(test_data_pt, test_model_pt)
        assert prob == 1.

    def test_inf_dist(self):
        """
        Tests the probability for a datapoint infinitely far from the model
        """
        test_data_pt = np.array([np.inf, np.inf, np.inf])
        test_model_pt = np.zeros(3)

        prob = self.model.point_point_prob(test_data_pt, test_model_pt)
        assert prob == 0.

    def test_neg_inf_dist(self):
        """
        Tests the probability for a datapoint at a distance -inf from the model
        """
        test_data_pt = np.array([-np.inf, -np.inf, -np.inf])
        test_model_pt = np.zeros(3)

        prob = self.model.point_point_prob(test_data_pt, test_model_pt)
        assert prob == 0.


class TestPointModelProbability(object):
    """
    Test the point to orbit model probability model.
    """

    def setup_method(self):
        """
        Set up the :obj:`Model` object for testing with a random fake dataset.
        """
        # The initialization data for this set of tests doesn't matter at all
        self.model = model.Model(np.random.rand(6, 3),
                                 np.random.rand(5, 2))
        self.test_model = np.zeros((6, 3))

    def test_inf_dist(self):
        """
        Tests the posterior probability for a datapoint at infinity in all axes
        """
        test_data_pt = np.array([np.inf, np.inf, np.inf])

        prob = self.model.point_model_prob(test_data_pt, self.test_model)
        assert prob == 0.

    def test_neg_inf_dist(self):
        """
        Tests the posterior probability for a datapoint at -inf in all axes
        """
        test_data_pt = np.array([-np.inf, -np.inf, -np.inf])

        prob = self.model.point_model_prob(test_data_pt, self.test_model)
        assert prob == 0.

    def test_no_dist(self):
        """
        Tests the posterior probability for a datapoint on the model
        """
        test_data_pt = np.array([0., 0., 0.])

        prob = self.model.point_model_prob(test_data_pt, self.test_model)
        assert prob == 1.

    def test_inf_model(self):
        test_data_pt = np.array([0., 0., 0.])

        test_model = np.array([[0., 0., 0.],
                               [np.inf, np.inf, np.inf],
                               [-np.inf, np.inf, np.inf],
                               [np.inf, -np.inf, np.inf],
                               [np.inf, np.inf, -np.inf],
                               [np.inf, -np.inf, -np.inf],
                               [-np.inf, np.inf, -np.inf],
                               [-np.inf, -np.inf, np.inf],
                               [-np.inf, -np.inf, -np.inf]])

        prob = self.model.point_model_prob(test_data_pt, test_model)
        assert prob == (1. / test_model.shape[0])


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

    def test_all_bound(self):
        params = np.random.rand(5)
        ln_prior = self.model.ln_prior(params)

        assert ln_prior == np.log(2**(-5))

    def test_all_unbound(self):
        params = np.random.rand(5) + 1
        ln_prior = self.model.ln_prior(params)

        assert ln_prior == -np.inf

    def test_mix_bound_unbound(self):
        params = np.random.rand(5)
        params[0] += 1
        params[2] += 1
        ln_prior = self.model.ln_prior(params)

        assert ln_prior == -np.inf


#class TestLnLike(object):
#    """
#    Test the natural log of our pdf
#    """
#
#    def test_inf_dist_data(self):
#        test_data = np.array(5 * [3 * [np.inf]])
#        pspace = np.array([[0., 1.],
#                           [0., 1.],
#                           [0., 1.],
#                           [0., 1.],
#                           [0., 1.]])
#
#        self.model = model.Model(test_data, pspace)
#        ln_prob = self.model.ln_like(np.random.rand(5))
#
#        assert ln_prob == -np.inf
