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
import json
import pickle
import yaml

parser = ArgumentParser(
    description='Apply the collaborative-ranking approach called GRank to a '
    'dataset of Amazon reviews.',
    epilog='Input file should contain a dictionary like:\n'
    '\t{"descriptions": {\n'
    '\t\t"<asin>": "<description of the item>",\n'
    '\t\t  ...       ...\n'
    '\t\t},\n'
    '\t "reviews": {\n'
    '\t\t"<asin>": {\n'
    '\t\t\t"<reviewerID>": <1 to 5 stars>,\n'
    '\t\t\t  ...            ...\n'
    '\t\t\t},\n'
    '\t\t  ...\n'
    '\t\t}\n'
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
    raise SystemExit('ERROR: input file format not supported!\n'
                     f'{parser.format_help()}')

extension = args.input.split('.')[-1]
if extension == 'json':
    with open(args.input, 'r') as f:
        data = json.load(f, parse_int=int)
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
