#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
    This script is part of a suite aimed to do item recommendations.

    Copyright (C) 2019 Federico Motta <191685@studenti.unimore.it>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
from numpy import array, quantile
from numpy.random import RandomState
from os import environ
from sklearn.model_selection import StratifiedShuffleSplit
from sys import stderr, version_info
import json
import pickle
import yaml


def debug(msg):
    """print msg on stderr"""
    print(msg, file=stderr)


class Dataset(dict):
    """import the input file and expose an interface of it"""

    @property
    def documents(self):
        """dictionary of <str document_id>: <str document_description>"""
        assert 'descriptions' in self, \
            'Dataset constructor did not initialize descriptions'
        return self['descriptions']

    @property
    def test_set(self):
        """dictionary of <str user_id>: [
               (<str document_id>, <bool like/dislike>), ... ]
        """
        assert 'test_set' in self, \
            'Dataset constructor did not build test set'
        return self['test_set']

    @property
    def training_set(self):
        """dictionary of <str user_id>: [
               (<str document_id>, <bool like/dislike>), ... ]
        """
        assert 'training_set' in self, \
            'Dataset constructor did not build training set'
        return self['training_set']

    def __init__(self, input_file):
        allowed_input = [
            ('.json', json, input_file, json.JSONDecodeError),
            ('<stdin>', json, input_file, json.JSONDecodeError),
            ('.pickle', pickle, input_file.buffer, pickle.PickleError),
            ('.yaml', yaml, input_file, (yaml.YAMLError, UnicodeDecodeError)),
        ]
        for extension, module, file_object, exceptions in allowed_input:
            if args.input.name.endswith(extension):
                try:
                    data = module.load(file_object)
                except exceptions as e:
                    parser.error(str(e))
                else:
                    break  # input file successfully loaded
        else:
            raise SystemExit('Input file format not supported, please use: '
                             '.json, .pickle or .yaml file.')

        # populate documents property
        self['descriptions'] = dict()
        for asin, description in data['descriptions'].items():
            self['descriptions'][asin] = description

        # merge test and training set because, in the content-based
        # part, a stratified-random-sampling (which will be done
        # afterwards) maintains the distribution of the user ratings,
        # while a simple random sampling does not.
        #
        # source: http://dl.acm.org/citation.cfm?id=295240.295795
        # (look at section "Experiments and results")
        self['users'] = dict()
        for source in ('test_set', 'training_set'):
            for asin, reviews in data[source].items():
                if asin not in self['descriptions'].keys():
                    debug(f'WARNING: document {asin} has no description')
                    continue

                for star, users in reviews.items():
                    for user in users:
                        if user not in self['users']:
                            self['users'][user] = list()

                        # Since a user rating criteria can be quite
                        # constant (e.g. a user who always rates 5
                        # stars) let us make it more variable by
                        # adding some noise in order to divide better
                        # training and test set afterwards
                        star = float(star) + next(NormalNoise(user))

                        self['users'][user].append(tuple((asin, star)))
                        debug(f'user {user:16} rated {star:.3f}/5 '
                              f'document {asin:16}')

        # populate training and test set properties with a stratified
        # random sampling
        self['test_set'] = dict()
        self['training_set'] = dict()
        for user, data in self['users'].items():
            top_quartile = quantile(a=[star for _, star in data], q=0.75)

            x, y = zip(*data)  # x == asin; y == star
            x = array(x)
            y = array([j >= top_quartile for j in y])
            train_idx, test_idx = next(
                StratifiedShuffleSplit(test_size=1 / 10,
                                       random_state=0).split(x, y))
            self['training_set'][user] = list(zip(x[train_idx], y[train_idx]))
            self['test_set'][user] = list(zip(x[test_idx], y[test_idx]))


class NormalNoise(object):
    """store in a class variable a personalized RandomState for each user"""

    _mu = 0  # mean
    _sigma = 1 / 3  # std-dev
    _generator = dict()  # per-user RandomState dictionary

    def __init__(self, user_id):
        if user_id not in NormalNoise._generator:
            # In order to have reproducible results, the random
            # generator is initialized with some user-dependant
            # information
            NormalNoise._generator[user_id] = RandomState(
                # the python builtin hash() uses a random salt;
                # setting the environment variable PYTHONHASHSEED
                # makes its returned values constant across different
                # runs
                seed=abs(hash(user_id)) % 2**32)
        self.user_id = user_id

    def __next__(self):
        """return a float between -1 and +1

           By using a std-dev of 1/3 and a mean of 1, the 99.7% of the
           values will be within mu ± 3 * std_dev; i.e. -1 and +1.

           Source: section "Standard_deviation_and_coverage" of
           https://en.wikipedia.org/wiki/Normal_distribution
        """
        return NormalNoise._generator[self.user_id].normal(
            self._mu, self._sigma)


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')
assert version_info >= (3, 6), 'Please use at least Python 3.6'

if str(environ.get('PYTHONHASHSEED', 'unset')) != '0':
    raise SystemExit('Please set environment variable PYTHONHASHSEED '
                     'to 0 (zero) to have reproducible results')

parser = ArgumentParser(description='\n\t'.join(
    ('', 'Apply the content-based approach called Naive Bayes to a dataset',
     'of Amazon reviews.', '',
     'Input file should be in one of the following supported formats:',
     '\t.json, .pickle, .yaml', '', 'And it should contain a dictionary like:',
     '\t{"test_set": {', '\t\t"<asin>": {"5": <list of reviewerID>,',
     '\t\t           "4": <list of reviewerID>,',
     '\t\t           "3": <list of reviewerID>,',
     '\t\t           "2": <list of reviewerID>,',
     '\t\t           "1": <list of reviewerID>},', '\t\t  ...', '\t\t},',
     '\t"training_set": {', '\t\t"<asin>": {"5": <list of reviewerID>,',
     '\t\t           "4": <list of reviewerID>,',
     '\t\t           "3": <list of reviewerID>,',
     '\t\t           "2": <list of reviewerID>,',
     '\t\t           "1": <list of reviewerID>},', '\t\t  ...', '\t\t},',
     '\t"descriptions": {"<asin>": "description of the item",',
     '\t                   ...      ...', '\t\t}', '\t}')),
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument(help='See the above input file specs.',
                    dest='input',
                    metavar='input_file',
                    type=FileType())
args = parser.parse_args()

Dataset(args.input).test_set
