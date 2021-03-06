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

from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter, SUPPRESS
from collections import Counter
from html import unescape
from math import log2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import array, log, mean, power, quantile, seterr, sqrt, zeros
from numpy.random import RandomState
from os import environ
from re import sub as regex_substitute
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB as MultinomialNaiveBayes
from sys import stderr, stdout, version_info
from string import punctuation
from tabulate import tabulate
import json
import pickle
import yaml


def accuracy(contingency_table):
    """(true positives + true negatives) / total cases

       actually sklearn returns a contingency_table like:
        true_negatives | false_positives
       ----------------+----------------
       false_negatives | true_positives
    """
    return float(contingency_table[1, 1] +
                 contingency_table[0, 0]) / contingency_table.sum()


def boolean_array(float_array, threshold):
    """return an array with True in positions above threshold"""
    assert isinstance(float_array, type(array([]))), \
        'First argument should be a numpy array'
    return float_array > threshold


def ascii_confusion_matrix(contingency_table):
    """return ascii-table representation of a contingency table"""
    return 'Confusion matrix:\n' + tabulate(
        [['', 'Liked\n(test-set)', 'Disliked\n(test-set)'],
         [
             'Liked\n(prediction)',
             int(contingency_table[1, 1]),
             int(contingency_table[1, 0])
         ],
         [
             'Disliked\n(prediction)',
             int(contingency_table[0, 1]),
             int(contingency_table[0, 0])
         ]],
        numalign='center',
        stralign='center',
        tablefmt='grid')


def debug(msg='', min_level=1, max_level=99):
    """print messages on stderr"""
    if args.loglevel and args.loglevel in range(min_level, max_level + 1):
        print(msg, file=stderr)


def f_score(contingency_table, beta=1):
    """return the harmonic average of precision and recall (with beta=1)"""
    return ((1 + power(beta, 2)) * precision(contingency_table) *
            recall(contingency_table)) / (
                (power(beta, 2) * precision(contingency_table)) +
                recall(contingency_table))


def get_X_y(user, preferences, corpus, scaling_coefficient):
    """Return matrix with document features and vector of ratings"""
    X, y = lil_matrix((len(preferences), len(corpus.T))), list()

    for row, (document, rating) in enumerate(preferences):
        X[row] = corpus.vector_of(document)
        y.append(round(rating * scaling_coefficient))
    return X.tocsr(), array(y, dtype=int)


def info(msg=''):
    print(msg, file=stderr)


def normalized_discount_cumulative_gain(rated_items, top_k_recommendations):
    """compute accuracy of the top-k recommended items"""

    user_mean_rate = round(5 * mean([round(rate) for _, rate in rated_items]))
    rating = {asin: round(5 * rate) for asin, rate in rated_items}
    for item, predicted_rating in top_k_recommendations:
        if item not in rating:
            rating[item] = user_mean_rate

    discounted_cumulative_gain = float(
        sum((pow(2, rating[item]) - 1) / log2(i + 2)  # because i starts from 0
            for i, (item, _) in enumerate(top_k_recommendations)))
    ideal_discounted_cumulative_gain = float(
        sum((pow(2, 5) - 1) / log2(i + 2)  # because i starts from 0
            for i, (item, _) in enumerate(top_k_recommendations)))
    return discounted_cumulative_gain / ideal_discounted_cumulative_gain


def precision(contingency_table):
    """true positives / (true positives + false positives)

       actually sklearn returns a contingency_table like:
        true_negatives | false_positives
       ----------------+----------------
       false_negatives | true_positives
    """
    old_settings = seterr(invalid='raise')
    try:
        return contingency_table[1, 1] / float(
            contingency_table.sum(axis=0)[1])
    except FloatingPointError as e:
        if float(contingency_table.sum(axis=0)[1]) < 1:
            debug('division by zero avoided, return Nan', min_level=2)
        else:
            debug(str(e), min_level=2)
        return float('nan')
    finally:
        seterr(**old_settings)


def recall(contingency_table):
    """true positives / (true positives + false negatives)

       actually sklearn returns a contingency_table like:
        true_negatives | false_positives
       ----------------+----------------
       false_negatives | true_positives
    """
    old_settings = seterr(invalid='raise')
    try:
        return contingency_table[1, 1] / float(
            contingency_table.sum(axis=1)[1])
    except FloatingPointError as e:
        if float(contingency_table.sum(axis=1)[1]) < 1:
            debug('division by zero avoided, return Nan', min_level=2)
        else:
            debug(str(e), min_level=2)
        return float('nan')
    finally:
        seterr(**old_settings)


class Corpus(dict):
    """apply standard natural language processing operations to the
       documents of a Dataset

       corpus_object[<str document_id>] = {
           'tokens' = ( token1, token2, ... ),
       }
    """

    dictionary_expansion = tuple(
        sorted(
            ('007', '2', '2002', '3', '360', '4', '64', '7', '9', 'action',
             'adaptor', 'adventure', 'alien', 'amazon', 'arkham', 'assassin',
             'asylum', 'av', 'basketball', 'batman', 'best', 'black', 'blu',
             'blue', 'book', 'brilliant', 'bundle', 'burnout', 'cable',
             'carbon', 'card', 'cd', 'century', 'change', 'charge',
             'comfortable', 'comic', 'controller', 'creed', 'criminal', 'data',
             'dead', 'deceptive', 'department', 'deserve', 'die', 'disc',
             'dreamcast', 'duke', 'dvd', 'eli', 'eng', 'entertain', 'evil',
             'expander', 'expansion', 'extra', 'fan', 'fifa', 'fun', 'game',
             'gamecube', 'genesis', 'german', 'globe', 'god', 'gold', 'good',
             'government', 'grand', 'gray', 'green', 'guitar', 'hard',
             'hardcore', 'hawk', 'hdtv', 'headset', 'hedgehog', 'hero',
             'hilarious', 'hope', 'humor', 'include', 'independent',
             'institute', 'interview', 'invasion', 'inventor', 'investigate',
             'kid', 'kit', 'kombact', 'like', 'logo', 'loyal', 'luck', 'man',
             'mario', 'marvel', 'mathematics', 'memory', 'mexico', 'microsoft',
             'mirror', 'modern', 'mortal', 'multi', 'murder', 'musicality',
             'nbsp', 'need', 'nintendo', 'nonfiction', 'officer', 'package',
             'pad', 'party', 'pc', 'pennyroyal', 'platform', 'play', 'player',
             'police', 'praise', 'product', 'professor', 'ps2', 'ps3', 'psp',
             'quality', 'quantum', 'ray', 'rechargeable', 'remote', 'reporter',
             'resident', 'revenge', 'romantic', 'sega', 'sennheiser', 'series',
             'sims', 'soccer', 'soft', 'solace', 'sony', 'speed', 'squaresoft',
             'station', 'super', 'tale', 'tooth', 'transfer', 'tribal', 'trip',
             'true', 'twentieth', 'ubi', 'vista', 'war', 'warmhearted', 'wii',
             'win', 'wireless', 'world', 'xbox'),
            key=len,
            reverse=True))

    T = tuple()  # sorted tuple with all the words in the documents

    @property
    def documents(self):
        """generator of document_id in the corpus"""
        for asin in sorted(self.keys()):
            yield asin

    @property
    def N(self):
        """number of documents in the corpus"""
        return len(self.keys())

    def __init__(self, dataset, do_lemmatization=True):
        assert isinstance(dataset, Dataset), \
            'Corpus constructor needs an instance of Dataset as argument'

        porter = PorterStemmer()
        stop_words = stopwords.words('english')
        word_net_lemmatizer = WordNetLemmatizer()

        Corpus.T = set()  # use a set temporarily
        for asin, description in dataset.documents.items():
            # unescape html character entity references
            raw_text = unescape(description)
            raw_text = raw_text.replace('’', "'")  # utf8 to ascii

            # split camel case words
            raw_text = regex_substitute(
                '([A-Z][a-z]+)', r' \1',
                regex_substitute('([A-Z]+)', r' \1', raw_text))

            # substitute any punctuation mark with a space
            raw_text = raw_text.translate(
                str.maketrans(punctuation, ' ' * len(punctuation)))

            # remove multiple spaces
            raw_text = ' '.join(raw_text.split())

            # split raw text into tokens
            token_list = word_tokenize(raw_text.lower(), language='english')

            # remove stop-words
            token_list = [
                token for token in token_list
                if token.lower() not in stop_words
            ]

            if do_lemmatization:  # do stemming of dictionary known words
                lemmas_list = list()
                for token in token_list:
                    # let us start with the hardcoded dictionary expansion
                    # since it is probably faster
                    for dict_word in Corpus.dictionary_expansion:
                        if dict_word.lower() in token.lower():
                            lemmas_list.append(dict_word)
                            break
                    else:
                        # then if any word matched, let us try with the
                        # slower word-net dictionary lemmatizer
                        for pos in ('n', 'a', 'v'):
                            lemma = word_net_lemmatizer.lemmatize(token,
                                                                  pos=pos)
                            if lemma != token:
                                lemmas_list.append(lemma)
                                break
                if lemmas_list:
                    token_list = lemmas_list  # update token list
                else:
                    debug(f'WARNING: empty token list after lemmatization\n'
                          f'\t(asin: {asin},\n'
                          f'\t token_list: {repr(token_list)},\n'
                          f'\t description: {repr(description)})')

            # do stemming of both words in and not in dictionary
            token_list = [porter.stem(token) for token in token_list]

            # remove tokens which have punctuaction
            token_list = [
                token for token in token_list
                if not set(token).intersection(punctuation) and len(token) > 2
            ]

            # let us ensure that the token list has at least a token
            # in order to avoid divisions by zero afterwards. This
            # empty token is added to all the documents in order to
            # make it not really important
            token_list.append('')

            # store and count the amount of tokens
            self[asin] = Counter(token_list)
            # precompute the total number of tokens in the document
            setattr(self[asin], 'number_of_tokens', sum(self[asin].values()))
            Corpus.T |= set(token_list)  # update all words in the corpus
        Corpus.T = tuple(sorted(Corpus.T))

        debug(f'"dictionary length" = |T| = {len(Corpus.T):7}')
        debug(f'"corpus length"     = |D| = {self.N:7}')
        self._cached_max_f = dict()
        self._cached_idf = dict()
        self._cached_denormalized_tf_idf = dict()
        self._cached_normalized_tf_idf = dict()
        self._cached_vector = dict()
        debug()

    def n(self, t_k):
        """return how many documents have token t_k at least once"""
        return sum(self[d_j][t_k] > 0 for d_j in self.documents)

    def f(self, t_k, d_j):
        """return the frequency of token t_k in document d_j"""
        return float(self[d_j][t_k]) / self[d_j].number_of_tokens

    def max_f(self, d_j):
        """return the maximum token frequency in document d_j"""
        try:  # try to return the cached value
            return self._cached_max_f[d_j]
        except KeyError:  # this is a cache miss, let us compute the value
            self._cached_max_f[d_j] = float(max(
                self[d_j].values())) / self[d_j].number_of_tokens
        return self._cached_max_f[d_j]

    def tf(self, t_k, d_j):
        """return normalized term frequency of token t_k in document d_j"""
        return self.f(t_k, d_j) / self.max_f(d_j)

    def idf(self, t_k):
        """return the inverse document frequency of term t_k in the corpus"""
        try:  # try to return the cached value
            return self._cached_idf[t_k]
        except KeyError:  # this is a cache miss, let us compute the value
            self._cached_idf[t_k] = log(self.N / self.n(t_k))
        return self._cached_idf[t_k]

    def _tf_idf(self, t_k, d_j):
        """return the denormalized TermFrequency-InverseDocumentFrequency"""
        try:  # try to return the cached value
            return self._cached_denormalized_tf_idf[(t_k, d_j)]
        except KeyError:  # this is a cache miss, let us compute the value
            self._cached_denormalized_tf_idf[(
                t_k, d_j)] = self.tf(t_k, d_j) * self.idf(t_k)  # TF * IDF
        return self._cached_denormalized_tf_idf[(t_k, d_j)]

    def cosine_normalized_tf_idf(self, t_k, d_j):
        """return the normalized TermFrequency-InverseDocumentFrequency"""
        if t_k not in self[d_j]:
            return 0  # because the TermFrequency factor would be zero
        try:  # try to return the cached value
            return self._cached_normalized_tf_idf[(t_k, d_j)]
        except KeyError:  # this is a cache miss, let us compute the value
            denominator = sqrt(
                sum(power(self._tf_idf(t_s, d_j), 2) for t_s in Corpus.T))
            if denominator < 1e-16:
                return 0  # avoid division-by-zero
            else:
                self._cached_normalized_tf_idf[(
                    t_k, d_j)] = self._tf_idf(t_k, d_j) / denominator
        return self._cached_normalized_tf_idf[(t_k, d_j)]

    tf_idf = cosine_normalized_tf_idf  # alias

    def vector_of(self, document_id):
        """return the vector with the weights of each token in the document"""
        try:  # try to return the cached value
            return self._cached_vector[document_id]
        except KeyError:
            self._cached_vector[document_id] = array([
                self.tf_idf(token, document_id)
                for k, token in enumerate(Corpus.T)
            ])
        return self._cached_vector[document_id]


class Dataset(dict):
    """import the input file and expose an interface of it"""

    @property
    def documents(self):
        """dictionary of <str document_id>: <str document_description>"""
        assert 'descriptions' in self, \
            'Dataset constructor did not initialize descriptions'
        return self['descriptions']

    @property
    def top_quartile(self):
        """dictionary of <str user_id>: <float between 0 and 1>"""
        assert 'top_quartile' in self, \
            'Dataset constructor did not build top_quartile'
        return self['top_quartile']

    @property
    def test_set(self):
        """dictionary of <str user_id>: [
               (<str document_id>, <float between 0 and 1>), ... ]
        """
        assert 'test_set' in self, \
            'Dataset constructor did not build test set'
        return self['test_set']

    @property
    def training_set(self):
        """dictionary of <str user_id>: [
               (<str document_id>, <float between 0 and 1>), ... ]
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
                        # normalize star value
                        # (x - x_min) / (x_max - x_min)
                        star = round(float(star + 1) / 7.0, ndigits=3)

                        self['users'][user].append(tuple((asin, star)))
                        debug(
                            f'user {user:16} rated {5 * star:.1f}/5 '
                            f'document {asin:16}',
                            min_level=2)

        # populate top_quartile, training and test set properties
        self['top_quartile'] = dict()
        self['test_set'] = dict()
        self['training_set'] = dict()
        for user, data in self['users'].items():
            top_quartile = quantile(a=[star for _, star in data],
                                    q=args.quantile_threshold)
            self['top_quartile'][user] = top_quartile

            x, y = zip(*data)  # x == asin; y == star
            x = array(x)
            if args.random:
                random_sampling = ShuffleSplit
            else:
                random_sampling = StratifiedShuffleSplit
            train_idx, test_idx = next(
                random_sampling(n_splits=2,
                                test_size=args.test_set_size,
                                random_state=0).split(
                                    x, array([j > top_quartile for j in y])))
            y = array(y)
            self['training_set'][user] = list(zip(x[train_idx], y[train_idx]))
            self['test_set'][user] = list(zip(x[test_idx], y[test_idx]))
        debug()


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

           In any case the 0.3% of the values outside [-1, +1] will be
           forced to the closest endpoint
        """
        return min(
            +1,
            max(
                -1, NormalNoise._generator[self.user_id].normal(
                    self._mu, self._sigma)))


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')
assert version_info >= (3, 6), 'Please use at least Python 3.6'

if str(environ.get('PYTHONHASHSEED', 'unset')) != '0':
    raise SystemExit('Please set environment variable PYTHONHASHSEED '
                     'to 0 (zero) to have reproducible results')

try:
    # use the Dumper from the compiled C library (if present)
    # because it is faster than the one for the python iterpreter
    yaml_dumper = yaml.CDumper
except AttributeError:
    yaml_dumper = yaml.Dumper  # fallback interpreted and slower Dumper
yaml_kwargs = dict(Dumper=yaml_dumper, default_flow_style=False)

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
parser.add_argument(
    '-v',
    '--verbose',
    action='count',
    default=0,
    dest='loglevel',
    help='print verbose messages (multiple -v increase verbosty)\n')
parser.add_argument('--random', action='store_true', help=SUPPRESS)
parser.add_argument('--confusion-matrix',
                    default=None,
                    help=SUPPRESS,
                    type=FileType('w'))
parser.add_argument('-s',
                    '--stop-after',
                    default=None,
                    help='stop script after doing recommendations for a '
                    'certain number of users',
                    metavar='int',
                    type=int)
parser.add_argument('--test-set-size',
                    default=0.1,
                    help='percentage of the ratings to put in the test set '
                    '(default: 0.10)',
                    metavar='float',
                    type=float)
parser.add_argument('-q',
                    '--quantile-threshold',
                    default=0.75,  # 4 / 5 # 4 stars out of 5
                    help='consider as liked items with a rating above this '
                    f'quantile threshold (default: 0.75)',
                    metavar='float',
                    type=float)
parser.add_argument('-o',
                    '--output',
                    default=None,
                    help='print script results on output file',
                    metavar='output_file',
                    type=FileType('w'))
parser.add_argument('-k',
                    '--top-k',
                    action='append',
                    dest='top_k',
                    help='compute also NDCG@k',
                    metavar='int')
parser.add_argument(
    '-a',
    '--scaling-factor',
    default=1e3,
    help='votes will be normalized and then scaled by that factor '
    '(default: 1000)',
    metavar='int',
    type=int)
parser.add_argument(
    '-t',
    '--threshold',
    default=None,
    help='use a greedy approach and stop after the first top-k '
    'suggestions above the threshold',
    metavar='float',
    type=float)

args = parser.parse_args()

if args.stop_after is not None and args.stop_after < 1:
    parser.error('-s/--stop-after must be greater than one')
if args.scaling_factor < 5:
    parser.error('-a/--scaling-factor must be at least 5')
if args.quantile_threshold < 0.01 or args.quantile_threshold > 0.99:
    parser.error('-q/--quantile-threshold must be in (0.01, 0.99)')
if args.test_set_size < 1 / 30 or args.test_set_size > 29 / 30:
    parser.error(f'--test-set-size must be in ({1/30:.2f}, {29/30:.2f})')
if args.threshold is not None and (args.threshold <= 0.01
                                   or args.threshold >= 0.99):
    parser.error('-t/--threshold must be in (0.01, 0.99)')

try:
    # ensure all the k are positive integers sorted by decreasing values
    args.top_k = [10] if not args.top_k else [int(k) for k in args.top_k]
    args.top_k = sorted(set([k for k in args.top_k if k > 0]), reverse=True)
except ValueError as e:
    if 'invalid literal for int' in str(e):
        parser.error('-k/--top-k only takes integers values')
    raise SystemExit(f'ERROR: {str(e)}')

dataset = Dataset(args.input)
corpus = Corpus(dataset)

# write command line arguments to output file (improves reproducibility)
output_header = ' Command Line Arguments '.center(80, '#') + '\n'
output_header += yaml.dump(
    {
        'args.input': args.input.name,
        'args.only_new': True,
        'args.output': getattr(args.output, 'name', None),
        'args.quantile_threshold': args.quantile_threshold,
        'args.scaling_factor': args.scaling_factor,
        'args.stop_after': args.stop_after,
        'args.test_set_size': args.test_set_size,
        'args.threshold': args.threshold,
        'args.top_k': args.top_k,
    }, **yaml_kwargs)
output_header += ' Results '.center(80, '#') + '\n'
if args.output is not None:
    args.output.write(output_header)
    args.output.flush()
    if args.output != stdout:
        args.output.close()  # close output file if != stdout

results = dict()  # collects results to save on output file
contingency_table = zeros((2, 2))
for i, (user, preferences) in enumerate(
        sorted(dataset.training_set.items(), key=lambda t: t[0])):
    if args.stop_after is not None and i >= args.stop_after:
        break

    # training
    classifier = MultinomialNaiveBayes()
    classifier.fit(*get_X_y(
        user, preferences, corpus, scaling_coefficient=args.scaling_factor))

    if user not in results:
        results[user] = dict(ranking=dict(), ndcg=dict())

    # prediction
    never_seen_documents = sorted(
        set(corpus.documents).difference(
            set(rated_doc for rated_doc, rate in preferences)))
    if args.threshold is None:
        # let us find the best recommendations in the whole corpus
        y_predicted = classifier.predict(
            csr_matrix(
                [corpus.vector_of(asin) for asin in never_seen_documents]))
        recommendations = zip(never_seen_documents, list(y_predicted))
    else:
        # let us find only the first top-k recommendations above the
        # threshold, which is in (0.01, 0.99)
        recommendations = dict()
        for asin in never_seen_documents:
            if len([
                    rating for rating in recommendations.values()
                    if rating >= args.threshold * args.scaling_factor
            ]) >= max(args.top_k):
                break
            recommendations[asin] = classifier.predict(
                csr_matrix([
                    corpus.vector_of(asin),
                ]))[0]
        recommendations = recommendations.items()
    top_k_recommendations = sorted(recommendations,
                                   key=lambda t: t[1],
                                   reverse=True)
    for k in args.top_k:
        info(f' Recommended items for user {user} '.center(80, '=') + '\n')
        for position, (item, rating) in enumerate(top_k_recommendations[:k]):
            rating /= float(args.scaling_factor)  # normalize predicted rating
            info(f'{position:>4d}) NaiveBayes(<item {item}>) {rating:>24.6f}')
            if position not in results[user]['ranking']:
                results[user]['ranking'][position] = dict(item=item,
                                                          rating=float(rating))

        rated_items = dataset.training_set[user] + dataset.test_set[user]
        ndcg = normalized_discount_cumulative_gain(rated_items,
                                                   top_k_recommendations[:k])
        info(f'\n{" " * 6}NDCG@{k}'.ljust(15) + f'{ndcg:>46.6f}\n')
        results[user]['ndcg'][k] = ndcg

    # rewrite output file with also this user results
    if args.output is not None and args.output != stdout:
        with open(args.output.name, 'w') as f:
            f.write(output_header + yaml.dump(results, **yaml_kwargs))

    # validation
    X_test, y_real = get_X_y(user,
                             dataset.test_set[user],
                             corpus,
                             scaling_coefficient=args.scaling_factor)
    y_predicted = classifier.predict(X_test)
    top_quartile = dataset['top_quartile'][user] * args.scaling_factor
    contingency_table += confusion_matrix(
        boolean_array(y_real, top_quartile),
        boolean_array(y_predicted, top_quartile))

# for each k compute the mean of all user's ndcg@k
ndcg_mean = {
    k: float(mean([user_data['ndcg'][k] for _, user_data in results.items()]))
    for k in args.top_k
}
# then rewrite output file with also this piece of information (ndcg mean)
if args.output is not None and args.output != stdout:
    with open(args.output.name, 'w') as f:
        f.write(output_header + yaml.dump(results, **yaml_kwargs) +
                ' Mean NDCG@K '.center(80, '#') + '\n' +
                yaml.dump(ndcg_mean, **yaml_kwargs) +
                ' Confusion matrix '.center(80, '#') + '\n#' +
                f'\n#{" " * 17}' + f'\n#{" " * 17}'.join(
                    ascii_confusion_matrix(contingency_table).split('\n')[1:]))

if args.confusion_matrix is not None:
    args.confusion_matrix.write(
        yaml.dump(
            {
                'args.input': args.input.name,
                'confusion_matrix': contingency_table.tolist(),
            }, **yaml_kwargs))

# and write it also on stdout
info(' Mean NDCG@K '.center(80, '=') + '\n')
for k in reversed(args.top_k):
    info(f'{" " *6}NDCG@{k}'.ljust(15) + f'{ndcg_mean[k]:>46.6f}')
info()

info(ascii_confusion_matrix(contingency_table) + '\n')
info(f'accuracy:  {accuracy(contingency_table):8.3f}')
info(f'precision: {precision(contingency_table):8.3f}')
info(f'recall:    {recall(contingency_table):8.3f}')
info(f'f-score:   {f_score(contingency_table):8.3f}')
