#!/usr/bin/env python
# 
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scorer.bleu_scorer import BleuScorer
import numpy as np

class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        # self._hypo_for_image = {}
        # self.ref_for_image = {}
        self.bleu_scorer = BleuScorer(n=self._n)

    def compute_score(self, gts, res):

        self.bleu_scorer.clear()
        for i, hypo in enumerate(res):
            ref = gts[i]

            # Sanity check.
            # assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            # self.bleu_scorer += (hypo[0], ref)
            self.bleu_scorer += (hypo, ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = self.bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score[-1], np.array(scores[-1])  # BLEU-4

    def method(self):
        return "Bleu"