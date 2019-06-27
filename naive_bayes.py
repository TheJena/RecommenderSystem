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
from collections import Counter
from html import unescape
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import array, log, power, quantile, sqrt
from numpy.random import RandomState
from os import environ
from re import sub as regex_substitute
from sklearn.model_selection import StratifiedShuffleSplit
from sys import stderr, version_info
from string import punctuation
import json
import pickle
import yaml


def debug(msg):
    """print msg on stderr"""
    print(msg, file=stderr)


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
        debug('')

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
        """dictionary of <str user_id>: <float between -1 and 6>"""
        assert 'top_quartile' in self, \
            'Dataset constructor did not build top_quartile'
        return self['top_quartile']

    @property
    def test_set(self):
        """dictionary of <str user_id>: [
               (<str document_id>, <bool like/dislike>), ... ]
        """
        assert 'test_set' in self, \
            'Dataset constructor did not build test set'
        return self['test_set']

    @property
    def training_set(self):
        """dictionary of <str user_id>: [
               (<str document_id>, <bool like/dislike>), ... ]
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

                        self['users'][user].append(tuple((asin, star)))
                        debug(f'user {user:16} rated {star:.3f}/5 '
                              f'document {asin:16}')

        # populate top_quartile, training and test set properties
        self['top_quartile'] = dict()
        self['test_set'] = dict()
        self['training_set'] = dict()
        for user, data in self['users'].items():
            top_quartile = quantile(a=[star for _, star in data], q=0.75)
            self['top_quartile'][user] = top_quartile

            x, y = zip(*data)  # x == asin; y == star
            x = array(x)
            y = array([j >= top_quartile for j in y])
            train_idx, test_idx = next(  # use a stratified random sampling
                StratifiedShuffleSplit(test_size=1 / 10,
                                       random_state=0).split(x, y))
            self['training_set'][user] = list(zip(x[train_idx], y[train_idx]))
            self['test_set'][user] = list(zip(x[test_idx], y[test_idx]))
        debug('')


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
        """
        return NormalNoise._generator[self.user_id].normal(
            self._mu, self._sigma)


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')
assert version_info >= (3, 6), 'Please use at least Python 3.6'

if str(environ.get('PYTHONHASHSEED', 'unset')) != '0':
    raise SystemExit('Please set environment variable PYTHONHASHSEED '
                     'to 0 (zero) to have reproducible results')

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
args = parser.parse_args()
dataset = Dataset(args.input)
corpus = Corpus(dataset)
