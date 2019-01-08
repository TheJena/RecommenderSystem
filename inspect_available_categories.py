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

from argparse import ArgumentParser
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from tabulate import tabulate

parser = ArgumentParser(description='Print a table with available '
                        'categories and their number of items')
parser.add_argument('-u',
                    '--user',
                    default=None,
                    dest='usr',
                    help='username to access the database',
                    metavar='str',
                    type=str)
parser.add_argument('-w',
                    '--pwd',
                    default=None,
                    dest='pwd',
                    help='password to access the database',
                    metavar='str',
                    type=str)
parser.add_argument('-a',
                    '--addr',
                    default='localhost',
                    help='MongoDB server address (default: localhost)',
                    metavar='str',
                    type=str)
parser.add_argument('-p',
                    '--port',
                    default=27017,
                    help='MongoDB server port (default: 27017)',
                    metavar='int',
                    type=int)
parser.add_argument('-d',
                    '--db',
                    default='test',
                    help='database name (default: test)\n ',
                    metavar='str',
                    type=str)
parser.add_argument('-c',
                    '--min-item-count',
                    default=1,
                    help='Minimum number of item for each category',
                    metavar='int',
                    type=int)
args = parser.parse_args()

# start a connection to the mongo database (with authentication if provided)
user_pwd = f'{args.usr}:{args.pwd}@' if args.usr and args.pwd else ''
uri = f'mongodb://{user_pwd}{args.addr}:{args.port}/?authSource={args.db}'
client = MongoClient(uri)

if False and tuple(
        int(n)
        for n in client.server_info()['version'].split('.')) < (3, 4, 0):
    raise SystemExit(
        '\nPlease upgrade your MongoDB instance to at least version 3.4.0\n'
        'otherwise some API operations, like $sortByCount aggregations\n'
        'can not be used\n')
db = client[args.db]
try:
    # check with a dummy query if the connection has been
    # successfully established.  If this did not happen,
    # the raisen OperationFailure exception is properly
    # caught and managed
    next(db.reviews.find().limit(1))  # this could raise OperationFailure
except OperationFailure as e:
    # the dummy query failed
    if 'authentication failed' in str(e).lower():
        # is it because of a wrong username or password?
        raise SystemExit('\nERROR: wrong username or password.\n')
    if any(('requires authentication' in str(e).lower(),
            'not authorized' in str(e).lower())):
        # or because authentication was required
        # and the user did not provide it?
        raise SystemExit('\nERROR: please authenticate to MongoDB with '
                         f'--user and --pwd\n\n'
                         '(with --help you can find other options)\n')
    # the dummy query failed because of some other reasons;
    # let us print them to the user and exit
    raise SystemExit(f'\nERROR: {str(e)}\n')

meta, reviews = db.meta, db.reviews

table = [(category, meta.count_documents({'categories': category}))
         for category in meta.distinct('categories')]
table = [(cat, num) for cat, num in table if num >= args.min_item_count]
table = sorted(table, key=lambda t: t[1], reverse=True)

print(f'Table of categories with at least {args.min_item_count} items:')
print(tabulate(table, headers=('category', 'n° items'), tablefmt='psql'))
