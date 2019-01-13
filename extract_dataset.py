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
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from textwrap import indent, wrap
import json
import pickle
import yaml


def indented_wrapped_repr(iterable):
    return indent('\n'.join(wrap(repr([c.replace(' ', '_')
                                       for c in iterable])[1:-1])),
                  '\t').replace('_', ' ')


parser = ArgumentParser(
    description='Extract amazon reviews from a dataset on MongoDB.',
    epilog='Output files contains a dictionary like:\n'
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
output_fmts = ('json', 'pickle', 'yaml')
parser.add_argument('-l', '--category-list',
                    action='store_true',
                    help='show available categories and exit\n ')
parser.add_argument('-a', '--addr',
                    default='localhost',
                    help='MongoDB server address',
                    metavar='str',
                    type=str)
parser.add_argument('-p', '--port',
                    default=27017,
                    help='MongoDB server port',
                    metavar='int',
                    type=int)
parser.add_argument('-u', '--user',
                    default=None,
                    dest='usr',
                    help='username to access the database',
                    metavar='str',
                    type=str)
parser.add_argument('-w', '--pwd',
                    default=None,
                    dest='pwd',
                    help='password to access the database',
                    metavar='str',
                    type=str)
parser.add_argument('-d', '--db',
                    default='test',
                    help='database name\n ',
                    metavar='str',
                    type=str)
parser.add_argument('-c', '--category',
                    action='append',
                    dest='categories',
                    help='extract items from the given category',
                    metavar='str')
parser.add_argument('-N', '--max-items',
                    default=5 * 10**4,
                    help='maximum number of items to extract',
                    metavar='int',
                    type=int)
parser.add_argument('-M', '--max-users',
                    default=7 * 10**5,
                    help='maximum number of users to extract',
                    metavar='int',
                    type=int)
parser.add_argument('-r', '--min-reviews',
                    default=1,
                    help='ignore users with less than min-reviews',
                    metavar='int',
                    type=int)
parser.add_argument('-R', '--max-reviews',
                    default=2 * 10**4,
                    help='ignore users with more than max-reviews\n ',
                    metavar='int',
                    type=int)
parser.add_argument('-o', '--output',
                    default=None,
                    dest='out',
                    help='dump extracted dataset to .'
                    f'{", .".join(output_fmts)} file.',
                    metavar='file',
                    type=str)
args = parser.parse_args()
user_pwd = f'{args.usr}:{args.pwd}@' if args.usr and args.pwd else ''
uri = f'mongodb://{user_pwd}{args.addr}:{args.port}/?authSource={args.db}'
client = MongoClient(uri)

try:
    if tuple(int(n) for n in client.server_info()['version'].split('.')
             ) < (3, 4, 0):
        raise SystemExit('\n'
            'Please upgrade your MongoDB instance to at least version 3.4.0\n'
            'otherwise some API operations, like $sortByCount aggregations\n'
            'can not be used\n')

    db = client[args.db]

    next(db.reviews.find().limit(1))  # could raise OperationFailure
    if args.category_list:
        print('Available categories:\n'
              f'{indented_wrapped_repr(set(db.meta.distinct("categories")))}.')
        raise SystemExit()
    elif args.out is not None and args.out.split('.')[-1] not in output_fmts:
        raise SystemExit('ERROR: output file format not supported!\n\n'
                         f'{parser.format_help()}')
except OperationFailure as e:
    if 'authentication failed' in str(e).lower():
        raise SystemExit('\nERROR: wrong username or password.\n')
    if any(('requires authentication' in str(e).lower(),
            'not authorized' in str(e).lower())):
        raise SystemExit('\nERROR: please authenticate to MongoDB with '
                         f'--user and --pwd\n\n'
                         '(with --help you can find other options)\n')
    raise SystemExit(f'\nERROR: {str(e)}\n')


args.categories = ['Video Games'] if not args.categories else args.categories
print(f'Using categories:\n{indented_wrapped_repr(args.categories)}.\n')

allowed_items = [d['asin'] for d in db.meta.find(
    {'categories': {'$in': args.categories}},
    {'_id': 0, 'asin': 1})]

all_users = {str(d['_id']): dict() for d in db.reviews.aggregate(
    [{'$match': {'asin': {'$in': allowed_items}}},
     {'$sortByCount': '$reviewerID'},
     {'$match': {'count': {'$gte': args.min_reviews,
                           '$lte': args.max_reviews}}},
     {'$limit': args.max_users}],
    allowDiskUse=True)}

items = set()
users = set()
num_reviews = 0
reviews = dict()
for user in all_users:
    query = ({'reviewerID': user, 'asin': {'$in': allowed_items}},
             {'_id': 0, 'asin': 1, 'overall': 1})
    for d in db.reviews.find(*query):
        asin, stars = str(d['asin']), str(int(d['overall']))
        if len(items) < args.max_items or asin in items:
            if asin not in reviews:
                reviews[asin] = dict()
            if stars not in reviews[asin]:
                reviews[asin][stars] = list()
            reviews[asin][stars].append(user)
            num_reviews += 1
            users.add(user)
            items.add(asin)

print(f'n° items: {len(items): >11}')
print(f'n° users: {len(users): >11}')
print(f'n° reviews: {num_reviews: >9}')

if args.out is None:
    raise SystemExit()
elif not items or not users:
    raise SystemExit('ERROR: too few items or users; no output file produced')

for a in items:
    reviews[a]['description'] = next(
        db.meta.find({'asin': a}, {'_id': 0, 'description': 1}))['description']

extension = args.out.split('.')[-1]
if extension == 'json':
    with open(args.out, 'w') as f:
        json.dump(reviews, f, indent=' ' * 4)
elif extension == 'pickle':
    with open(args.out, 'wb') as f:
        pickle.dump(reviews, f, protocol=pickle.HIGHEST_PROTOCOL)
elif extension == 'yaml':
    try:
        dumper = yaml.CDumper
    except AttributeError:
        dumper = yaml.Dumper
    with open(args.out, 'w') as f:
        yaml.dump(reviews, f, default_flow_style=False, Dumper=dumper)
