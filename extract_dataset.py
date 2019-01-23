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
from queue import Queue
from textwrap import indent, wrap
from threading import Lock, Thread
import json
import pickle
import yaml


def indented_wrapped_repr(iterable):
    """return iterable representation (str) wrapped in 80 columns
       and indented with one tab
    """
    return indent('\n'.join(wrap(repr([c.replace(' ', '_')
                                       for c in iterable])[1:-1])),
                  '\t').replace('_', ' ')


parser = ArgumentParser(
    description='Extract amazon reviews from a dataset on MongoDB.',
    epilog='Output files contains a dictionary like:\n'
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
    formatter_class=RawTextHelpFormatter)
output_fmts = ('json', 'pickle', 'yaml')  # allowed output formats
parser.add_argument('-l', '--category-list',
                    action='store_true',
                    help='show available categories and exit\n ')
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
parser.add_argument('-a', '--addr',
                    default='localhost',
                    help='MongoDB server address                '
                    '(default: localhost)',
                    metavar='str',
                    type=str)
parser.add_argument('-p', '--port',
                    default=27017,
                    help='MongoDB server port                   '
                    '(default: 27017)',
                    metavar='int',
                    type=int)
parser.add_argument('-d', '--db',
                    default='test',
                    help='database name                         '
                    '(default: test)\n ',
                    metavar='str',
                    type=str)
parser.add_argument('-c', '--category',
                    action='append',
                    dest='categories',
                    help='extract items also from X category    '
                    '(default: "Video Games")',
                    metavar='X')
parser.add_argument('-R', '--max-review',
                    default=2 * 10**4,
                    help='ignore users with more than r reviews '
                    '(default: 20000)\n',
                    metavar='r',
                    type=int)
parser.add_argument('-M', '--users',
                    default=100,
                    help='extract m users                       '
                    '(default: 100)',
                    metavar='m',
                    type=int)
parser.add_argument('-T', '--reviews',
                    choices=(30, 40, 50, 60),
                    default=30,
                    help='extract t reviews for each user       '
                    '(default: 30)',
                    metavar='t',
                    type=int)
parser.add_argument('-j', '--threads',
                    default=8,
                    help='run n threads in parallel             '
                    '(default: 8)\n ',
                    metavar='n',
                    type=int)
parser.add_argument('-o', '--output',
                    default=None,
                    dest='out',
                    help='dump extracted dataset to file f\n'
                    f'(supported extensions: .{", .".join(output_fmts)})',
                    metavar='f',
                    type=str)
args = parser.parse_args()

# start a connection to the mongo database (with authentication if provided)
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

    # check with a dummy query if the connection has been
    # successfully established.  If this did not happen,
    # the raisen OperationFailure exception is properly
    # caught and managed
    next(db.reviews.find().limit(1))  # this could raise OperationFailure

    # discover the list of available categories if the -l/--category-list
    # option was provided
    if args.category_list:
        print('Available categories:\n'
              f'{indented_wrapped_repr(set(db.meta.distinct("categories")))}.')
        raise SystemExit()  # exit after the print of the availables categories

    # check that the output file has an extension compatible with the
    # allowed output formats; otherwise print an error message and exit
    elif args.out is not None and args.out.split('.')[-1] not in output_fmts:
        raise SystemExit('ERROR: output file format not supported!\n\n'
                         f'{parser.format_help()}')
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

# force the number of threads to use to be at least two; because this script
# implements a producer/consumer model which need at least one of both to work
args.threads = 2 if args.threads < 2 else args.threads

# choose the default category if no one was provided
args.categories = ['Video Games'] if not args.categories else args.categories

# inform the user about the category/categories that are going to be used
print(f'Using categories:\n{indented_wrapped_repr(args.categories)}.\n')

# allowed_items is a tuple of items ids, sorted by decreasing number of
# reviews
#
# this is achieved in two steps:
# 1) with the db.meta.find query; which creates a list of item ids.  Each of
#    these items is extracted from the collection meta and has at least one
#    category of those specified by the user (-c/--category)
# 2) with the db.reviews.aggregate query; which returns the tuple of item ids.
#    Each of these item is extracted from the collection reviews and is also
#    present in the list returned from step 1.  Then this intermediate result
#    is sorted by the number of reviews each item has, and truncated to the
#    maximum number of items this script could need: n° users * n° reviews
#    (which could happen if every user had reviewed different items; these two
#    bounds are defined by the user with -M/--users and -T/--reviews)
allowed_items = tuple(str(d['_id']) for d in db.reviews.aggregate(
    [{'$match': {'asin': {'$in': list(d['asin'] for d in db.meta.find(
        {'categories': {'$in': args.categories}},
        projection={'_id': False, 'asin': True}))}}},
     {'$sortByCount': '$asin'},
     {'$limit': args.users * args.reviews}],
    allowDiskUse=True,
    batchSize=args.reviews))

# let us initialize the definitive set of item we will use as an empty set
items = set()

# users is a tuple of user ids, which have reviewed the items in
# allowed_items and that are sorted by decreasing number of reviews.  Besides
# each of them has at least done T reviews, with T \in {30, 40, 50, 60}
# (defined by the user with -T/--reviews) and at most -R/--max-review.  The
# latter bound ensures that for example users which did more than 20'000
# reviews (and that are considered spammers) are not considered.  Finally the
# remaining tuple of user ids is truncated to -M/--users
users = tuple(str(d['_id']) for d in db.reviews.aggregate(
    [{'$match': {'asin': {'$in': allowed_items}}},
     {'$sortByCount': '$reviewerID'},
     {'$match': {'count': {'$gte': args.reviews,
                           '$lte': args.max_review}}},
     {'$limit': args.users}],
    allowDiskUse=True,
    batchSize=args.users))


def query_consumer():
    """Receive one by one user from the main thread through the queue
       query_queue and ask the mongo db if that user reviewed any of the items
       in allowed_items (from the most reviewed to less one); if anyone of
       these query is successful, the respective item is sent to the
       reviewer_consumer; otherwise it is skipped.  A new user is explored when
       the previous one has collected -T/--reviews reviews or has reached the
       end of allowed_items.  The first 10 collected items of each user are
       marked as items for the test set.
    """
    while True:
        user = query_queue.get()  # receive a user from the main thread
        if user is None:
            break
        i = 0
        for item in allowed_items:  # explore the allowed items
            if i >= args.reviews:
                # the user has collected -T/--reviews reviews;
                # let us move to the next user
                break
            # ask the db if the user has reviewed the current item
            d = next(db.reviews.find({'reviewerID': user, 'asin': item},
                                     projection={'_id': False,
                                                 'asin': True,
                                                 'overall': True,
                                                 'reviewerID': True},
                                     limit=1),
                     None)  # if this did not happen assign None to d
            if d is not None:
                # Well done, the user has reviewd the item; let us send the
                # query result to the review_consumer
                # (aka we are review producer)
                review_queue.put(dict(d=d, test_set=i<10))
                i += 1
        query_queue.task_done()


def review_consumer():
    """Receive from the query_consumer one by one dictionaries like:
       {'d':        {'asin': str,
                     'overall': str,
                     'reviewerID': str},
        'test_set': bool}
       and add these informations in the final object that will be serialized
       to the output file.
    """
    while True:
        obj = review_queue.get()  # receive a dictionary from query_consumer
        if obj is None:
            break
        # ensure only one thread as access to the global reviews object
        reviews_lock.acquire()
        # set object destination (test/training set)
        # accordingly to its boolean flag
        if obj['test_set']:
            data = reviews['test_set']
        else:
            data = reviews['training_set']
        d = obj['d']
        # extract item id, n° stars and user id
        asin, stars, user = d['asin'], int(d['overall']), d['reviewerID']
        asin, stars, user = str(asin), str(stars), str(user)
        if asin not in data:
            data[asin] = dict()
        if stars not in data[asin]:
            data[asin][stars] = list()
        # add to destination the information that:
        # the $user has reviewed the $item with $stars
        data[asin][stars].append(user)
        # add item to global items set
        items.add(asin)
        reviews_lock.release()
        review_queue.task_done()


# initialize data structures needed for the
# multithreaded producer/consumer model
reviews_lock, threads = Lock(), list()
query_queue, review_queue = Queue(), Queue()

# initialize the object to serialize to the output file
reviews = dict(test_set=dict(), training_set=dict(), descriptions=dict())

# start one review_consumer
# and (-j/--threads - 1) query_consumer (aka review_producer)
for i in range(args.threads):
    if i < 1:
        t = Thread(target=review_consumer)
    else:
        t = Thread(target=query_consumer)
    t.start()
    threads.append(t)

# send to query_consumer/review_producer the users one by one
# (actually the query_queue is filled in a buffer fashion)
for user in users:
    query_queue.put(user)

# wait until every object in the queues/buffers has been processed
query_queue.join()
review_queue.join()

# send (through the queues) to every thread a None object which means
# "good job, go in peace"
for i in range(args.threads):
    if i < 1:
        review_queue.put(None)
    else:
        query_queue.put(None)

# wait for each thread end
for t in threads:
    t.join()

# tell the user how many users/items/reviews were collected
print(f'n° users: {len(users): >11}')
print(f'n° items: {len(items): >11}')
print(f'n° reviews: {len(users) * args.reviews: >9}')

# if the user did not ask (-o/--output) to serialize the extracted informations
# it is time to exit
if args.out is None:
    raise SystemExit()

# otherwise let us collect also the descriptions of the item we collected
# this is achieved with a find query on the collection meta which returns
# for each item the description of the first item with the same id
for a in items:
    reviews['descriptions'][a] = next(
        db.meta.find({'asin': a},
                     projection={'_id': False,
                                 'description': True},
                     limit=1)
    )['description']

# let us serialize the extracted dataset in the different supported formats
# if you need a human readable output you could use the prettily indented
# yaml or json formats otherwise use a binary pickle one which is more compact
extension = args.out.split('.')[-1]
if extension == 'json':
    with open(args.out, 'w') as f:
        json.dump(reviews, f, indent=' ' * 4)
elif extension == 'pickle':
    with open(args.out, 'wb') as f:
        pickle.dump(reviews, f, protocol=pickle.HIGHEST_PROTOCOL)
elif extension == 'yaml':
    try:
        # use the Dumper from the compiled C library (if present)
        # because it is faster than the one for the python iterpreter
        dumper = yaml.CDumper
    except AttributeError:
        dumper = yaml.Dumper  # fallback interpreted and slower Dumper
    with open(args.out, 'w') as f:
        yaml.dump(reviews, f, default_flow_style=False, Dumper=dumper)
