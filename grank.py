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
from networkx.classes.function import degree
from numpy.linalg import norm
from numpy.random import RandomState
from scipy.sparse import dok_matrix
import json
import pickle
import yaml


class Item():
    """abstract concept of item, with its desirable and undesirable faces"""

    @property
    def d(self):
        return self._label + '_d'

    @property
    def u(self):
        return self._label + '_u'

    def __init__(self, label):
        self._label = str(label)

    def __eq__(self, other):
        return self._label == other._label

    def __hash__(self):
        return hash(self._label)

    def __str__(self):
        return self._label

class Preference():
    """desirable item is preferrable over undesirable item"""

    @property
    def desirable(self):
        """desirable Item"""
        return self._d

    @property
    def undesirable(self):
        """undesirable Item"""
        return self._u

    def __init__(self, desirable_item, undesirable_item):
        self._d = desirable_item
        self._u = undesirable_item

    def __eq__(self, other):
        return all((self.desirable.d == other.desirable.d,
                    self.undesirable.u == other.undesirable.u))

    def __hash__(self):
        return hash((self.desirable.d, self.undesirable.u))

    def __str__(self):
        return f'{self.desirable.d} > {self.undesirable.u}'


class Observation():
    """preference of a user between two reviewed items with different rating"""

    @property
    def user(self):
        """author of the preference"""
        return self._user

    def __init__(self, user, preference):
        self._user = str(user)
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
        """set of Items used to generate 3rd layer"""
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
            self._items.add(Item(d['item']))
        for d in test_set_reviews:
            self._users.add(d['user'])
            self._items.add(Item(d['item']))

        # create observation set
        comparisons = tuple(
            (S, s) for S in range(5, 1, -1) for s in range(S - 1, 0, -1))
        self._observations = set()
        for user in self.users:
            user_reviews = {s: set() for s in range(1, 6)}
            for d in (d for d in training_set_reviews if d['user'] == user):
                user_reviews[int(d['stars'])].add(d['item'])

            for more_stars, less_stars in comparisons:
                for asin_d in user_reviews[more_stars]:
                    for asin_u in user_reviews[less_stars]:
                        self._observations.add(
                            Observation(user, Preference(Item(asin_d),
                                                         Item(asin_u))))

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
                raise ValueError(
                    f'ERROR: user node "{obs.user}" not found in TPG')
            self.add_edge(obs.user, str(obs.preference))

            if obs.preference.desirable.d not in self.nodes():
                raise ValueError(
                    f'ERROR: desirable node "{obs.preference.desirable.d}"'
                    ' not found in TPG')

            if obs.preference.undesirable.u not in self.nodes():
                raise ValueError(
                    f'ERROR: undesirable node "{obs.preference.undesirable.u}"'
                    ' not found in TPG')


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
    def alpha(self):
        """damping factor"""
        return self._alpha

    @property
    def transition_matrix(self):
        """
           WARNING:
               the transition matrix has already been multiplied by alpha
        """
        return self._transition_matrix

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

        self._alpha = alpha

        t = dok_matrix((self.tpg.number_of_nodes(),
                        self.tpg.number_of_nodes()),
                       dtype=float)
        for a, b in self.tpg.edges:
            i, j = self.tpg.nodes[a]['id'], self.tpg.nodes[b]['id']
            t[i, j] = self.alpha / float(degree(self.tpg, a))
            t[j, i] = self.alpha / float(degree(self.tpg, b))
        self._transition_matrix = t.tocsr()

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

    def personalized_vector(self, user):
        pv = dok_matrix((self.tpg.number_of_nodes(), 1), dtype=float)
        pv[self.tpg.nodes[user]['id'], 0] = 1
        return pv.tocsc()

    def gr(self, item, rank_vector, min_prob=1e-16, max_prob=1):
        i_d, i_u = self.tpg.nodes[item.d]['id'], self.tpg.nodes[item.u]['id']
        PPR_id = min(max_prob, max(rank_vector[i_d, 0], min_prob))
        PPR_iu = min(max_prob, max(rank_vector[i_u, 0], min_prob))
        return PPR_id / (PPR_id + PPR_iu)

    def top_k_recommendations(self, user, k=10, max_iter=10**3):
        """return a dict with the top_k recommendations and some statistics"""
        max_iter = 10**3 if max_iter < 1 else max_iter
        ret = dict(delta_PPR=list(), k=k, max_iter=max_iter, user=user)
        PV = self.personalized_vector(user)
        PPR = RandomState(
            seed=abs(hash(user)) % 2**32).rand(self.tpg.number_of_nodes(), 1)
        PPR = PPR / PPR.sum()
        for it in range(1, max_iter):
            PPR_before = PPR
            PPR = self.transition_matrix.dot(PPR) + (1 - self.alpha) * PV
            delta_PPR = norm(PPR - PPR_before)
            ret['delta_PPR'].append(delta_PPR)
            if delta_PPR < 1e-16:
                ret['iterations'] = it
                break
        else:
            print(f'WARNING: PPR did not converge after {max_iter} iterations;'
                  ' stop forced')
            ret['iterations'] = max_iter
        ret['top_k'] = sorted([(str(i), self.gr(i, PPR))
                               for i in self.tpg.items],
                              key=lambda t: t[1],
                              reverse=True)[:k]
        return ret


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
