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

from argparse import ArgumentParser, RawTextHelpFormatter
from networkx import Graph
import json
import pickle
import yaml


class Preference():
    """desirable item is preferrable over undesirable item"""

    @property
    def d(self):
        """desirable item"""
        return self._d

    @property
    def u(self):
        """undesirable item"""
        return self._u

    def __init__(self, desirable_item, undesirable_item):
        self._d = str(desirable_item) + '_d'
        self._u = str(undesirable_item) + '_u'

    def __eq__(self, other):
        return all((self.d == other.d, self.u == other.u))

    def __hash__(self):
        return hash((self.d, self.u))

    def __str__(self):
        return f'{self.d} > {self.u}'


class Observation():
    """preference of a user between two reviewed items with different rating"""

    @property
    def user(self):
        """author of the preference"""
        return self._u

    def __init__(self, user, preference):
        self._u = str(user)
        self.preference = preference

    def __eq__(self, other):
        return all((self.user == other.user,
                    self.preference == other.preference))

    def __hash__(self):
        return hash((self.user, self.preference))

    def __str__(self):
        return f'{self.user} ~> ({self.preference})'


class TPG(Graph):
    """tripartite graph"""

    @property
    def users(self):
        """set of users corresponding to nodes in 1st layer"""
        return self._users

    @property
    def preferences(self):
        """set of preferences corresponding to nodes in 2nd layer"""
        return set(o.preference for o in self.observations)

    @property
    def items(self):
        """set of items used to generate 3rd layer"""
        return self._items

    @property
    def observations(self):
        """set of observ. corresponding to links between 1st and 2nd layer"""
        return self._observations

    @property
    def specs(self):
        return '\n'.join(('',
                          'Tripartite Graph specs:',
                          f'n° edges: {self.number_of_edges(): >21}',
                          f'n° nodes: {self.number_of_nodes(): >21}',
                          f'├── user layer: {len(self.users): >15}'
                          ' (both training and test set)',
                          f'├── preference layer: {len(self.preferences): >9}'
                          ' (only training          set)',
                          f'└── (un)desirable layer: {2 * len(self.items): >6}'
                          ' (both training and test set)',
                          ))

    def __init__(self, **kwargs):
        if any(('test_set_reviews' not in kwargs,
                'training_set_reviews' not in kwargs)):
            raise ValueError('ERROR: please set both training_set_reviews and '
                             'test_set_reviews kwargs in TPG constructor.')

        training_set_reviews = kwargs['training_set_reviews']
        test_set_reviews = kwargs['test_set_reviews']
        super().__init__()

        # create user and item sets
        self._users = set()
        self._items = set()
        for d in training_set_reviews:
            self._users.add(d['user'])
            self._items.add(d['item'])
        for d in test_set_reviews:
            self._users.add(d['user'])
            self._items.add(d['item'])

        # create observation set
        comparisons = tuple(
            (S, s) for S in range(5, 1, -1) for s in range(S - 1, 0, -1))
        self._observations = set()
        for u in self.users:
            user_reviews = {s: set() for s in range(1, 6)}
            for d in (d for d in training_set_reviews if d['user'] == u):
                user_reviews[int(d['stars'])].add(d['item'])

            for more_stars, less_stars in comparisons:
                for d_item in user_reviews[more_stars]:
                    for u_item in user_reviews[less_stars]:
                        self._observations.add(
                            Observation(u, Preference(d_item, u_item)))

        # build graph
        index = 0
        for user in self.users:
            self.add_node(user, id=index, type='user')
            index += 1

        for item in self.items:
            self.add_node(f'{item}_d', id=index, type='desirable')
            index += 1
            self.add_node(f'{item}_u', id=index, type='undesirable')
            index += 1

        for obs in self.observations:
            if str(obs.preference) not in self.nodes():
                self.add_node(str(obs.preference), id=index, type='preference')
                index += 1

            if obs.user not in self.nodes():
                raise ValueError('ERROR: user node '
                                 f'"{obs.user}" not found in TPG')
            self.add_edge(obs.user, str(obs.preference))

            if obs.preference.d not in self.nodes():
                raise ValueError('ERROR: desirable node '
                                 f'"{obs.preference.d}" not found in TPG')
            self.add_edge(str(obs.preference), obs.preference.d)

            if obs.preference.u not in self.nodes():
                raise ValueError('ERROR: undesirable node '
                                 f'"{obs.preference.u}" not found in TPG')
            self.add_edge(str(obs.preference), obs.preference.u)


class GRank():

    allowed_input_formats = ('json', 'pickle', 'yaml')

    @property
    def tpg(self):
        """tripartite graph"""
        return self._tpg

    @property
    def test_set(self):
        return self._test_set

    @property
    def training_set(self):
        return self._training_set

    @property
    def specs(self):
        return '\n'.join(
            ('',
             'Dataset specs:',
             f'n° users: {len(self.tpg.users): >21}'
             ' (both training and test set)',
             f'n° items: {len(self.tpg.items):21}'
             ' (both training and test set)',
             f'n° reviews: {len(self.reviews(training_set=True)): >19}'
             ' (only training          set)',
             f'n° observations: {len(self.tpg.observations): >14}'
             ' (only training          set)'))

    def __init__(self, file_object, alpha=0.85):
        extension = file_object.name.split('.')[-1]
        if extension not in self.allowed_input_formats:
            raise SystemExit('ERROR: input file format not supported, please '
                             f'use: .{", .".join(GRank.allowed_input_formats)}'
                             ' file.\n')
        elif extension == 'json':
            data = json.load(file_object)
        elif extension == 'pickle':
            data = pickle.load(open(file_object.name, 'rb'))
        elif extension == 'yaml':
            try:
                loader = yaml.CLoader
            except AttributeError:
                loader = yaml.Loader
            data = yaml.load(file_object, Loader=loader)
        try:
            self._test_set = data['test_set']
            self._training_set = data['training_set']
            self._descriptions = data['descriptions']
        except KeyError as e:
            print(f'ERROR: {str(e)}')
        else:
            print(f'Successfully loaded dataset from {file_object.name}')

        self._test_set_reviews = None
        self._training_set_reviews = None

        self._tpg = TPG(training_set_reviews=self.reviews(training_set=True),
                        test_set_reviews=self.reviews(test_set=True))

    def reviews(self, test_set=False, training_set=False):
        """cache and return a tuple of reviews"""
        if test_set and self._test_set_reviews is None:
            self._test_set_reviews = tuple(self._reviews(test_set=True))
        elif training_set and self._training_set_reviews is None:
            self._training_set_reviews = tuple(
                self._reviews(training_set=True))
        if test_set:
            return self._test_set_reviews
        elif training_set:
            return self._training_set_reviews
        raise ValueError('ERROR: please set test_set or training_set flag')

    def _reviews(self, test_set=False, training_set=False):
        """private reviews generator"""
        if test_set:
            data = self.test_set
        elif training_set:
            data = self.training_set
        else:
            raise ValueError('ERROR: please set test_set or training_set flag')
        for item, d in data.items():
            for stars, reviewers in d.items():
                if stars not in tuple(str(i) for i in range(1, 6)):
                    raise ValueError(f'Invalid stars: "{repr(stars)}"; '
                                     'string expected.')
                for user in reviewers:
                    yield dict(user=str(user), item=str(item), stars=int(stars))


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')

parser = ArgumentParser(description='Apply the collaborative-ranking approach '
                        'called GRank to a dataset of Amazon reviews.',
                        formatter_class=RawTextHelpFormatter)
parser.add_argument(help='load dataset from: '
                    f'.{", .".join(GRank.allowed_input_formats)} file.\n'
                    'It should contain a dictionary like:\n'
                    '\t{"test_set": {\n'
                    '\t\t"<asin>": {"5": <list of reviewerID>,\n'
                    '\t\t           "4": <list of reviewerID>,\n'
                    '\t\t           "3": <list of reviewerID>,\n'
                    '\t\t           "2": <list of reviewerID>,\n'
                    '\t\t           "1": <list of reviewerID>},\n'
                    '\t\t  ...\n'
                    '\t\t},\n'
                    '\t"training_set": {\n'
                    '\t\t"<asin>": {"5": <list of reviewerID>,\n'
                    '\t\t           "4": <list of reviewerID>,\n'
                    '\t\t           "3": <list of reviewerID>,\n'
                    '\t\t           "2": <list of reviewerID>,\n'
                    '\t\t           "1": <list of reviewerID>},\n'
                    '\t\t  ...\n'
                    '\t\t},\n'
                    '\t"descriptions": {"<asin>": "description of the item",\n'
                    '\t                   ...      ...\n'
                    '\t\t}\n'
                    '\t}',
                    dest='input',
                    metavar='input_file',
                    type=open)
args = parser.parse_args()
grank = GRank(args.input)
print(grank.specs)
print(grank.tpg.specs)
