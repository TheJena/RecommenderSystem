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
from sys import version_info
import yaml


def ndcg_coefficient(user_data):
    """return coefficient to weight the recommendations in user_data dict"""
    try:
        return user_data['ndcg'][args.top_k]
    except KeyError:
        for k, ndcg in sorted(user_data['ndcg'].items(), key=lambda t: t[0]):
            if k >= args.top_k:
                return ndcg  # ndcg of the k nearest to top-k
        return min(user_data['ndcg'].values())


def parse_yaml_file(file_object, num_dict=3, prefix='#', suffix='#\n'):
    """parse yaml file_object and return a tuple with num_dict objects"""
    ret = list()
    tmp_buffer = ''
    for line in file_object.readlines():
        if line.startswith(prefix) and line.endswith(suffix):  # separator
            if tmp_buffer:  # parse buffer and flush it
                ret.append(yaml.load(tmp_buffer, Loader=yaml_loader))
                tmp_buffer = ''
            continue  # waste current line
        if not line.startswith('#'):
            tmp_buffer += line  # add line to buffer
    if tmp_buffer:
        ret.append(yaml.load(tmp_buffer, Loader=yaml_loader))
    assert len(ret) == num_dict, str(
        f'Expected number of dictionaries ({num_dict}) in file '
        f'{file_object.name} differs from the real one {len(ret)}')
    return tuple(ret)


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')
assert version_info >= (3, 6), 'Please use at least Python 3.6'

try:
    # use the Loader from the compiled C library (if present)
    # because it is faster than the one for the python iterpreter
    yaml_loader = yaml.CLoader
except AttributeError:
    yaml_loader = yaml.Loader  # fallback interpreted and slower Loader

parser = ArgumentParser(description='\n\t'.join(
    ('', 'Merge results from two different approaches to do',
     'recommendations:', '    - content-based (Naive Bayes)',
     '    - collaborative-ranking (GRank)')),
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('-c',
                    '--content-based',
                    help='file with suggestions from content-based approach',
                    metavar='yaml',
                    required=True,
                    type=FileType())
parser.add_argument('-g',
                    '--graph-based',
                    help='file with suggestions from graph-based approach',
                    metavar='yaml',
                    required=True,
                    type=FileType())
parser.add_argument('-s',
                    '--stop-after',
                    default=None,
                    help='stop script after a certain number of users',
                    metavar='int',
                    type=int)
parser.add_argument('-k',
                    '--top-k',
                    default=10,
                    help='number of items to recommend',
                    metavar='int',
                    type=int)
args = parser.parse_args()

if args.stop_after is not None and args.stop_after < 1:
    parser.error('-s/--stop-after must be greater than one')
if args.top_k < 1:
    parser.error('-k/--top-k must be greater or equal to one')

content_based = parse_yaml_file(args.content_based)
graph_based = parse_yaml_file(args.graph_based)

if content_based[0]['args.input'] != graph_based[0]['args.input']:
    parser.error('--content-based and --graph-based results must have the '
                 'same dataset in "args.input" field')
if content_based[0]['args.only_new'] != graph_based[0]['args.only_new']:
    parser.error('--content-based and --graph-based results must have the '
                 'same boolean value in "args.only_new" field')
if not all(('args.scaling_factor' in content_based[0],
            'args.test_set_size' in content_based[0])):
    parser.error('-c/--content-based file must have '
                 'args.scaling_factor and args.test_set_size fields')
if not all(
    ('args.max_iter' in graph_based[0], 'args.threshold' in graph_based[0])):
    parser.error('-g/--graph-based file must have '
                 'args.max_iter and args.threshold fields')

for i, user_in_common in enumerate(
        sorted(
            set(graph_based[1].keys()).intersection(
                set(content_based[1].keys())))):
    if args.stop_after is not None and i >= args.stop_after:
        break

    naive_bayes = content_based[1][user_in_common]
    nb_coeff = ndcg_coefficient(naive_bayes)

    grank = graph_based[1][user_in_common]
    gr_coeff = ndcg_coefficient(grank)

    # put in recommendations dict the linear combination of the
    # ranking of the two approaches:
    #     naive_bayes_ndcg * naive_bayes_ranking  +  grank_ndcg * grank_ranking
    recommendations = dict()
    for data in naive_bayes['ranking'].values():
        recommendations[data['item']] = data['rating'] * nb_coeff

    for data in grank['ranking'].values():
        if data['item'] not in recommendations:
            recommendations[data['item']] = 0
        if 'rating' in data:
            recommendations[data['item']] += data['rating'] * gr_coeff

    col = ('position', 'item', 'weighted rating')
    lines = [
        f' Recommendations for user "{user_in_common}" '.center(80, '='), '',
        f'naive bayes weight coefficient: {nb_coeff:.3f}',
        f'      grank weight coefficient: {gr_coeff:.3f}', '',
        f'{col[0].center(9)} | {col[1].center(16)} | {col[2].center(16)}',
        f'{"-" * 9}-+-{"-" * 16}-+-{"-" * 16}'
    ]

    for position, (item, rating) in enumerate(
            sorted(recommendations.items(), key=lambda t: t[1], reverse=True)):
        if position + 1 > args.top_k:
            break
        lines.append(f'{position + 1:^9} | {item:^16} | {rating:^16.3f}')
    else:
        lines.append(f'no more item available{" " * 7}')
    lines.append(f'{"-" * 9}-+-{"-" * 16}-+-{"-" * 16}')
    lines.append('')

    print('\n'.join((line.center(80).rstrip() for line in lines)))
