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


parser = ArgumentParser(
    description='Apply the collaborative-ranking approach called GRank to a '
    'dataset of Amazon reviews.',
    epilog='Input file should contain a dictionary like:\n'
    '\t{"<asin>": {\n'
    '\t\t"description": "<description of the item>",\n'
    '\t\t"5":            <list of reviewerID>,\n'
    '\t\t"4":            <list of reviewerID>,\n'
    '\t\t"3":            <list of reviewerID>,\n'
    '\t\t"2":            <list of reviewerID>,\n'
    '\t\t"1":            <list of reviewerID>\n'
    '\t\t},\n'
    '\t\t...\n'
    '\t }\n ',
    formatter_class=RawTextHelpFormatter)
input_fmts = ('json', 'pickle', 'yaml')
parser.add_argument('-i', '--input',
                    default=None,
                    help=f'load dataset from .{", .".join(input_fmts)} file.',
                    metavar='file',
                    required=True,
                    type=str)
args = parser.parse_args()

if args.input is not None and args.input.split('.')[-1] not in input_fmts:
    raise SystemExit('ERROR: input file format not supported!\n\n'
                     f'{parser.format_help()}')

extension = args.input.split('.')[-1]
if extension == 'json':
    with open(args.input, 'r') as f:
        data = json.load(f)
elif extension == 'pickle':
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
elif extension == 'yaml':
    try:
        loader = yaml.CLoader
    except AttributeError:
        loader = yaml.Loader
    with open(args.input, 'r') as f:
        data = yaml.load(f, Loader=loader)

items = set(data.keys())
print(f'n° items: {len(items): >15}')

reviews, users = list(), set()
for i, d in data.items():
    for k, v in d.items():
        if k == 'description':
            continue
        for u in v:
            reviews.append(dict(user=str(u), item=str(i), stars=int(k)))
            users.add(u)
print(f'n° users: {len(users): >15}')
print(f'n° reviews: {len(reviews): >13}')

if not items or not users:
    raise SystemExit('ERROR: too few items or users')

observations = list()
for user in users:
    user_reviews = {s: [] for s in range(1, 6)}
    for d in reviews:
        if d['user'] != user:
            continue
        user_reviews[d['stars']].append(d['item'])
    for high_stars in range(5, 1, -1):
        for low_stars in range(high_stars - 1, 0, -1):
            for d_item in user_reviews[high_stars]:
                for u_item in user_reviews[low_stars]:
                    observations.append(
                        Observation(user, Preference(d_item, u_item)))
print(f'n° preferences: {len(observations): > 9}')

tpg = Graph()
for obs in observations:
    tpg.add_node(obs.user, type='user')
    tpg.add_node(obs.preference.d, type='desirable')
    tpg.add_node(obs.preference.u, type='undesirable')
    tpg.add_node(str(obs.preference), type='preference')

    tpg.add_edge(obs.user, str(obs.preference))
    tpg.add_edge(str(obs.preference), obs.preference.d)
    tpg.add_edge(str(obs.preference), obs.preference.u)
