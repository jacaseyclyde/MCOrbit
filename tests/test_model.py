#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:32:12 2018

@author: jacaseyclyde
"""
# pylint: disable=C0413
import unittest

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np  # noqa

import mcorbit  # noqa


class TestPointPointProbability(unittest.TestCase):
    """
    Test the point to point probability model
    """

    def setUp(self):
        """
        Set up the :obj:`Model` object for testing with a random fake dataset.
        """
        self.model = mcorbit.model.Model(np.random.rand(6, 3))

    def test_point_point_prob_no_dist(self):
        """
        Tests the probability for a datapoint on top of the model point
        """
        test_data_pt = np.zeros(3)
        test_model_pt = np.zeros(3)

        prob = self.model.point_point_prob(test_data_pt, test_model_pt)
        self.assertEqual(prob, 1.)

    def test_point_point_prob_inf_dist(self):
        """
        Tests the probability for a datapoint infinitely far from the model
        """
        test_data_pt = np.array([np.inf, np.inf, np.inf])
        test_model_pt = np.zeros(3)

        prob = self.model.point_point_prob(test_data_pt, test_model_pt)
        self.assertEqual(prob, 0.)

    def test_point_point_prob_neg_inf_dist(self):
        """
        Tests the probability for a datapoint at a distance -inf from the model
        """
        test_data_pt = np.array([-np.inf, -np.inf, -np.inf])
        test_model_pt = np.zeros(3)

        prob = self.model.point_point_prob(test_data_pt, test_model_pt)
        self.assertEqual(prob, 0.)


if __name__ == '__main__':
    unittest.main()
