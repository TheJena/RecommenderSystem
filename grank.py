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

    @property
    def d(self):
        """desirable item"""
        return self._d

    @property
    def u(self):
        """undesirable item"""
        return self._u

    def __init__(self, desirable, undesirable):
        self._d = str(desirable) + '_d'
        self._u = str(undesirable) + '_u'

    def __str__(self):
        return f'{self.d} > {self.u}'


class Observation():

    @property
    def user(self):
        """author of the review"""
        return self._u

    def __init__(self, user, preference):
        self._u = str(user)
        self.preference = preference

    def __str__(self):
        return f'{self.user} ~> ({self.preference})'


def reviews_generator(data):
    for asin, d in data.items():
        for stars, reviewers in d.items():
            if stars not in tuple(str(i) for i in range(1, 6)):
                raise ValueError(f'Invalid stars: "{repr(stars)}"; '
                                 'string expected.')
            for user in reviewers:
                yield dict(user=str(user), item=str(asin), stars=int(stars))


parser = ArgumentParser(
    description='Apply the collaborative-ranking approach called GRank to a '
    'dataset of Amazon reviews.',
    formatter_class=RawTextHelpFormatter)
allowed_formats = ('json', 'pickle', 'yaml')
parser.add_argument(help='load dataset from: '
                    f'.{", .".join(allowed_formats)} file.\n'
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

if args.input.name.split('.')[-1] not in allowed_formats:
    raise SystemExit('ERROR: input file format not supported!\n\n'
                     f'{parser.format_help()}')

extension = args.input.name.split('.')[-1]
if extension == 'json':
    data = json.load(args.input)
elif extension == 'pickle':
    data = pickle.load(open(args.input.name, 'rb'))
elif extension == 'yaml':
    try:
        loader = yaml.CLoader
    except AttributeError:
        loader = yaml.Loader
    data = yaml.load(args.input, Loader=loader)

test_set_reviews = tuple(reviews_generator(data['test_set']))
training_set_reviews = tuple(reviews_generator(data['training_set']))

users = set(d['user'] for d in training_set_reviews)
users |= set(d['user'] for d in test_set_reviews)
print(f'n° users: {len(users): >15}'
      ' (both training and test set)')

items = set(d['item'] for d in training_set_reviews)
items |= set(d['item'] for d in test_set_reviews)
print(f'n° items: {len(items): >15}'
      ' (both training and test set)')

print(f'n° reviews: {len(training_set_reviews): >13}'
      ' (only training          set)')

comparisons = tuple(
    (S, s) for S in range(5, 1, -1) for s in range(S - 1, 0, -1))

observations = list()
for u in users:
    user_reviews = {s: set() for s in range(1, 6)}
    for d in (d for d in training_set_reviews if d['user'] == u):
        user_reviews[int(d['stars'])].add(d['item'])

    for more_stars, less_stars in comparisons:
        for d_item in user_reviews[more_stars]:
            for u_item in user_reviews[less_stars]:
                observations.append(Observation(u, Preference(d_item, u_item)))
print(f'n° preferences: {len(observations): > 9}'
      ' (only training          set)')

tpg = Graph()
for user in users:
    tpg.add_node(user, type='user')

for item in items:
    tpg.add_node(f'{item}_d', type='desirable')
    tpg.add_node(f'{item}_u', type='undesirable')

for obs in observations:
    tpg.add_node(str(obs.preference), type='preference')

    if obs.user not in tpg.nodes():
        raise ValueError(f'ERROR: user node "{obs.user}" not found in TPG')
    tpg.add_edge(obs.user, str(obs.preference))

    if obs.preference.d not in tpg.nodes():
        raise ValueError(
            f'ERROR: desirable node "{obs.preference.d}" not found in TPG')
    tpg.add_edge(str(obs.preference), obs.preference.d)

    if obs.preference.u not in tpg.nodes():
        raise ValueError(
            f'ERROR: undesirable node "{obs.preference.u}" not found in TPG')
    tpg.add_edge(str(obs.preference), obs.preference.u)
