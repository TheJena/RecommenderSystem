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
from math import log2
from multiprocessing import cpu_count, current_process, Lock, Process, Queue
from numpy import mean
from numpy.linalg import norm
from numpy.random import RandomState
from scipy.sparse import csr_matrix, dok_matrix
from signal import signal, SIGINT
from sys import float_info, stderr, stdout, version_info
import json
import pickle
import yaml


def info(msg, show=True):
    """flush msg to sterr if show flag is set"""
    if show:
        print(msg, end='', file=stderr, flush=True)


def process_body(input_queue, output_queue):
    """multiple processes spawned in background run this function.

       1) take a user from the input_queue
       2) compute top-k recommendations (and ndcg@k) over this user
       3) put results in output_queue
       4) goto 1
    """
    while True:
        target_user = input_queue.get()
        if target_user is None:  # caught no-more-work-message
            output_queue.put((None, None, None, None))  # let us ack it
            return
        for k in args.top_k:
            # if anybody is using stderr let us show some progress on it
            show = stderr_lock.acquire(block=False)
            output_queue.put(
                (target_user, k,
                 grank.top_k_recommendations(target_user, k, show),
                 grank.ndcg(target_user, k, show)))
            if show:
                stderr_lock.release()


def sigint_handler(sig_num, stack_frame):
    """Cleanily catch SIGINT and exit without printing long stack traces"""
    if sig_num != SIGINT.value:
        return
    if current_process().name == 'MainProcess':
        info(f'\n\nReceived SIGINT (Ctrl + C)\n\n')
    raise SystemExit(sig_num)


class Dataset(dict):
    """import the input file and expose an interface of it"""

    allowed_input_formats = ('json', 'pickle', 'yaml')

    @property
    def specs(self):
        both = ' (both training and test set)'
        train = ' (only training          set)'
        return '\n    '.join((
            f'\nDataset specs:',
            f'n° users {self.M:>34}{both}',
            f'n° items {self.N:>34}{both}',
            f'n° reviews '
            f'{len(tuple(r for r in self.reviews(training_set=True))):>32}'
            f'{train}\n',
        ))

    @property
    def test_set(self):
        """dictionary of <str item_id>: {<str n° star>: [<str user_id>, ...],
                                          ... }
        """
        return self['test_set']

    @property
    def training_set(self):
        """dictionary of <str item_id>: {<str n° star>: [<str user_id>, ...],
                                          ... }
        """
        return self['training_set']

    @property
    def descriptions(self):
        """dictionary of <str item_id>: <str item description>"""
        return self['descriptions']

    @property
    def users(self):
        """set of <str user_id>"""
        return set(u for source in (self.test_set, self.training_set)
                   for item, d in source.items() for stars, users in d.items()
                   for u in users)

    @property
    def items(self):
        """set of <str item_id>"""
        return set(item for source in (self.test_set, self.training_set)
                   for item in source)

    @property
    def M(self):
        """n° of users"""
        if getattr(self, '_m', None) is None:
            self._m = len(self.users)
        return self._m

    @property
    def N(self):
        """n° of items"""
        if getattr(self, '_n', None) is None:
            self._n = len(self.items)
        return self._n

    def __init__(self, file_object):
        # check that input file format is allowed, then load its data
        extension = file_object.name.split('.')[-1]
        if extension not in self.allowed_input_formats:
            raise SystemExit(
                'ERROR: input file format not supported, please use: .'
                f'{", .".join(Dataset.allowed_input_formats)} file.\n')
        elif extension == 'json':
            data = json.load(file_object)
        elif extension == 'pickle':
            data = pickle.load(open(file_object.name, 'rb'))
        elif extension == 'yaml':
            try:
                loader = yaml.CLoader  # faster compiled Loader
            except AttributeError:
                loader = yaml.Loader  # fallback, slower interpreted Loader
            data = yaml.load(file_object, Loader=loader)
        for key in ('test_set', 'training_set', 'descriptions'):
            if key not in data:
                raise SystemExit(
                    'ERROR: missing "{key}" in file {file_object.name}')
            else:
                self[key] = data[key]
        info(f'Successfully loaded dataset from {file_object.name}\n')

    def reviews(self, test_set=False, training_set=False):
        """reviews generator; a review is a dict like:
           {'user': <str user_id>, 'item': <str item_id>, 'stars': <int stars>}
        """
        if test_set and not training_set:
            source = self.test_set
        elif not test_set and training_set:
            source = self.training_set
        else:
            raise ValueError('ERROR: please set test_set or training_set flag')
        for item, d in source.items():
            for stars, reviewers in d.items():
                if stars not in tuple(str(i) for i in range(1, 6)):
                    raise ValueError(f'Invalid stars: "{repr(stars)}"; '
                                     f'string expected.')
                for user in reviewers:
                    yield dict(user=str(user),
                               item=str(item),
                               stars=int(stars))


class Item():
    """abstract concept of item, with its desirable and undesirable faces"""
    @property
    def d(self):
        """desirable item face"""
        return self._label + '_d'

    @property
    def u(self):
        """undesirable item face"""
        return self._label + '_u'

    def __init__(self, label):
        self._label = str(label)

    def __eq__(self, other):
        return self._label == other._label

    def __le__(self, other):
        return self._label <= other._label

    def __lt__(self, other):
        return self._label < other._label

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
        return all(
            (self.user == other.user, self.preference == other.preference))

    def __hash__(self):
        return hash((self.user, self.preference))

    def __str__(self):
        return f'{self.user} ~> ({self.preference})'


class TPG():
    """tripartite graph"""

    comparisons = tuple(
        (S, s) for S in range(5, 1, -1) for s in range(S - 1, 0, -1))

    @property
    def specs(self):
        both = ' (both training and test set)'
        train = ' (only training          set)'
        return '\n    '.join((
            f'\nTripartite Graph specs:',
            f'n° nodes: {self.number_of_nodes:>33}',
            f'├── U (users) {self.M:>29}{both}',
            f'├── P (preferences) '
            f'{len(self.preferences) + self.number_of_missing_preferences:>23}'
            f'{train}',
            f'└── R (representatives) {2 * self.N:>19}{both}',
            f'n° edges: {self.number_of_edges:>33}',
            f'├── f: U x P -> {"{0, 1}"} {len(self.observations):>20}'
            f' (agreement function)',
            f'└── s: P x R -> {"{0, 1}"} {2 * self.N * (self.N - 1):>20}'
            f' (support   function)\n',
        ))

    @property
    def dataset(self):
        assert isinstance(self._dataset, Dataset), \
            "TPG.dataset is not an instance of Dataset class"
        assert self._dataset, "TPG.dataset is empty"
        return self._dataset

    @property
    def users(self):
        """nodes in 1st layer

           dictionary of <str user_id>: <int user_degree>
        """
        assert isinstance(self._users, dict), "TPG.users is not a dictionary"
        assert self._users, "TPG.users is empty"
        return self._users

    @property
    def M(self):
        """number of nodes in 1st layer"""
        return len(self.users)

    @property
    def observations(self):
        """links between 1st and 2nd layer

           set of Observations
        """
        assert isinstance(self._observations,
                          set), "TPG.observations is not a set"
        assert self._observations, "TPG.observations is empty"
        return self._observations

    @property
    def preferences(self):
        """nodes in 2nd layer

           dictionary of <Preference pref_obj>: <int preference_degree>
           (only nodes connected to at least a user are considered here)
        """
        assert isinstance(self._preferences,
                          dict), "TPG.preferences is not a dictionary"
        assert self._preferences, "TPG.preferences is empty"
        return self._preferences

    @property
    def missing_preferences(self):
        """
           generate tuples like (Preference, <int index>)
           which corresponds to Preferences nodes in 2nd layer
           which were not chosen by any user
        """
        index = self.M + len(self.preferences)
        for desirable_item in self.items:
            for undesirable_item in self.items:
                if desirable_item != undesirable_item:
                    preference = Preference(desirable_item, undesirable_item)
                    # ensure this preference has not yet been considered in
                    # self.preferences
                    # (aka those one linked to at least a user)
                    if preference not in self.preferences:
                        yield (preference, index)
                        index += 1

    @property
    def number_of_missing_preferences(self):
        """number of nodes in 2nd layer which are not linked
           to any node in the 1st layer
        """
        return self.N * (self.N - 1) - len(self.preferences)

    @property
    def items(self):
        """tuple of Item, for each Item two nodes are generated in 3rd layer
           (one for the item desirable face and one for the undesirable one)
        """
        assert isinstance(self._items, tuple), "TPG.items is not a tuple"
        assert self._items, "TPG.items is empty"
        return self._items

    @property
    def N(self):
        """number of items (half of the nodes in 3rd layer)"""
        return len(self.items)

    @property
    def number_of_nodes(self):
        """number of nodes in 1st, 2nd and 3rd layer"""
        return self.M + self.N * (self.N - 1) + 2 * self.N

    @property
    def number_of_edges(self):
        """number of edges between 1st and 2nd layer
           and between 2nd and 3rd one
        """
        return len(self.observations) + 2 * self.N * (self.N - 1)

    def __init__(self, dataset):
        self._dataset = dataset  # initialize dataset property

        # populate Item set
        self._items = tuple(sorted(Item(item) for item in self.dataset.items))
        # create users nodes (with a null degree)
        self._users = {user: dict(degree=0) for user in self.dataset.users}

        # create preferences nodes, and populate observation set
        # (which corresponds to the edges between 1st and 2nd layer)
        # warning: only preferences explicitly chosen by at least a user are
        #          considered here; those linked only to the 3rd layer will
        #          be generated next (because they are too many to stay in ram)
        # user or preference node degree are also updated any time a new edge,
        # starting/ending from/to it, is hit
        self._preferences = dict()
        self._observations = set()
        for user in self.users:
            # create a dictionary with a key for each n° stars
            user_reviews = {s: set() for s in range(1, 6)}
            for d in self.dataset.reviews(training_set=True):
                if d['user'] == user:
                    # add any reviewed item to the corresponding n° stars key
                    user_reviews[int(d['stars'])].add(d['item'])

            # initialize the dictionary of Preferences ant the Observations set
            # by iterating over comparisons between items with more stars
            # against those with less stars (i.e. items with 5 stars vs items
            # with 4, ... 5 stars vs 1 star, 4 stars vs 3 stars, ...,
            # 2 stars vs 1)
            for more_stars, less_stars in self.comparisons:
                for asin_d in user_reviews[more_stars]:
                    for asin_u in user_reviews[less_stars]:
                        preference = Preference(Item(asin_d), Item(asin_u))
                        self._observations.add(Observation(user, preference))
                        if preference not in self._preferences:
                            self._preferences[preference] = dict(degree=3)
                            self._users[user]['degree'] += 1
                        else:
                            self._preferences[preference]['degree'] += 1
                            self._users[user]['degree'] += 1

        # assign indexes to graph nodes stored in memory
        # (those which would not fit into ram will be generated
        # with their corresponding indexes)
        index = 0
        for user, data in self.users.items():
            data['index'] = index
            index += 1
        for preference, data in self.preferences.items():
            data['index'] = index
            index += 1

    def missing_desirable_item_index(self, item):
        """return index of desirable item face"""
        if getattr(self, '_missing_desirables', None) is None:
            self._missing_desirables = dict()
            for index, item in enumerate(self.items):
                self._missing_desirables[item] = \
                    index + self.M + self.N * (self.N - 1)
        return self._missing_desirables[item]

    def missing_undesirable_item_index(self, item):
        """return index of undesirable item face"""
        return self.missing_desirable_item_index(item) + self.N


class GRank():
    """run grank recommendation algorithm on a tripartite graph"""
    @property
    def specs(self):
        return '\n    '.join(
            ('\nGRank specs:', f'max iterations {args.max_iter:>28d}',
             f'threshold{" " * 33}{args.threshold:g}',
             f'alpha {self.alpha:>40.2f}\n\n'))

    @property
    def tpg(self):
        """tripartite graph"""
        return self._tpg

    @property
    def alpha(self):
        """damping factor"""
        return self._alpha

    @property
    def transition_matrix_1(self):
        """transition matrix could be partitioned in the following blocks:

           users .  preferences  . items x2
                 |  1  |         |          <~ users
            -----+-----+---------+-----
              2  |     |         |  3       <~
            -----+-----+---------+-----     <~
                 |     |         |          <~ preferences
                 |     |         |  4       <~
                 |     |         |          <~
            -----+-----+---------+-----
                 |  5  |    6    |          <~ items x2

            This property memorizes the above transition matrix without values
            that would be in blocks 4 and 6.  In fact these two blocks sparsity
            together with their size make them impossible to fit in ram.
            Fortunately they are reproducible with a python generator because
            their structure is regular enough; in fact:
            - each row of block 4 has two non zero cells, which are distant N
            - each col of block 4 has N - 1 non zero cells
            - the mask of the block 4 is the transposed of the mask of block 6
            - block 4 can be represented by: 1/2 * mask(<block 4>)
            - block 6 can be represented by: 1/(N - 1) * mask(<block 6>)

            In order to do less multiplications afterwards the matris elements
            are already moltiplied by alpha; thus:

            cell[i, j] = alpha / degree(node i)
        """
        if getattr(self, '_transition_matrix_1', None) is None:
            show = current_process().name in ('MainProcess', 'Process-1')
            info(f'Building sparse transition matrix T:{" " * 23}', show)
            # to build this sparse matrix the usage of a Dictionary Of Keys
            # based one is really handy
            t1 = dok_matrix(
                (self.tpg.number_of_nodes, self.tpg.number_of_nodes))
            total = len(self.tpg.observations)
            # iterate over edges between 1st and 2nd layer
            for i, obs in enumerate(self.tpg.observations):
                if i % (total // 10**4) == 0:
                    info('\b' * 8 + f'{100 * i / total:>6.2f} %', show)

                user = self.tpg.users[obs.user]  # user node
                user_index, user_degree = map(user.get, ('index', 'degree'))

                pref = self.tpg.preferences[obs.preference]  # preference node
                pref_index, pref_degree = map(pref.get, ('index', 'degree'))

                item_d = obs.preference.desirable  # item desirable face
                item_u = obs.preference.undesirable  # item undesirable face
                item_d_index = self.tpg.missing_desirable_item_index(item_d)
                item_u_index = self.tpg.missing_undesirable_item_index(item_u)
                item_d_degree, item_u_degree = self.tpg.N - 1, self.tpg.N - 1

                # block 1
                t1[user_index, pref_index] = self.alpha / user_degree
                # block 2
                t1[pref_index, user_index] = self.alpha / pref_degree
                # block 3
                t1[pref_index, item_d_index] = self.alpha / pref_degree
                t1[pref_index, item_u_index] = self.alpha / pref_degree
                # block 5
                t1[item_d_index, pref_index] = self.alpha / item_d_degree
                t1[item_u_index, pref_index] = self.alpha / item_u_degree

            # use a Compressed Sparse Row matrix which performs more
            # efficiently matrix-vector multiplications
            self._transition_matrix_1 = t1.tocsr()
            info('\b' * 64 + 'Sparse transition matrix T specs:'.ljust(64),
                 show)
            # compute and show some informations
            nnz = self._transition_matrix_1.nnz \
                + self.tpg.number_of_missing_preferences * 2 * 2
            size = self.tpg.number_of_nodes**2
            density = 1 - ((size - nnz) / size)
            ram_size = self._transition_matrix_1.data.nbytes
            if ram_size < 1024**2:
                ram_size /= 1024
                ram_unit = 'kb'
            elif ram_size < 1024**3:
                ram_size /= 1024**2
                ram_unit = 'mb'
            else:
                ram_size /= 1024**3
                ram_unit = 'gb'
            info(
                '\n    '.join(
                    ('', f'n° of non zero elements {nnz:>26d}',
                     f'n° of all elements {size:>31d}',
                     f'density {density:>49.3g}',
                     f'size in ram {ram_size:>41.2f}  {ram_unit}\n')) + '\n',
                show)
        return self._transition_matrix_1

    def __init__(self, dataset, alpha=0.85):
        self._alpha = alpha
        self._dataset = dataset
        self._test_set_reviews = None
        self._training_set_reviews = None
        self._transition_matrix_1 = None
        self._recommendation_output = dict()

        # build a tripartite graph from the loaded dataset
        self._tpg = TPG(dataset)

    def alpha_dot_transition_matrix_dot(self, ppr, show):
        """return the result of:   alpha * T * PPR

           Please note that T is too large to fit into ram the above operation
           has been splitted into the next one:
                (alpha * T1 * PPR) + (alpha * T2 * PPR)

           Since the transition matrix T can be partitioned in blocks like:

           users .  preferences  . items x2
                 |  1  |         |          <~ users
            -----+-----+---------+-----
              2  |     |         |  3       <~
            -----+-----+---------+-----     <~
                 |     |         |          <~ preferences
                 |     |         |  4       <~
                 |     |         |          <~
            -----+-----+---------+-----
                 |  5  |    6    |          <~ items x2

           T1 is the transition matrix with only blocks: 1, 2, 3, 5
           T2 is the transition matrix with only blocks: 4, 6

           Since T1 can fit into ram, it is kept in memory; on the other side
           T2 which is too large and too sparse is generated line by line with
           python generators.

           Keep in mind that cell values are:   cell[i, j] = 1 / degree(i)
           Which means that block 4 has cells with 1/2 as value and block 6 has
           cells with 1/(N -1) as value.  Cells in other blocks have a variable
           degree accordingly to how many users expressed a given preference.
        """
        # initialize result as the first term of the sum:  alpha * T1 * PPR
        result = self.transition_matrix_1.dot(ppr)

        pref_value = float(self.alpha) / 2.0  # ... block 2
        item_value = float(self.alpha) / float(self.tpg.N - 1)  # ... block 6

        info(' ' * 38, show)
        total = self.tpg.number_of_missing_preferences
        # iterate over edges between 2nd and 3rd layer
        # (a python generator is used because they would not fit in ram)
        for i, (p, p_index) in enumerate(self.tpg.missing_preferences):
            assert i < total, f'ERROR: i > total ({i} > {total}'
            if i % (total // 10**4) == 0 and i / total < 1:
                info('\b' * 8 + f'{100 * i / total:>6.2f} %', show)

            d_index = self.tpg.missing_desirable_item_index(p.desirable)
            u_index = self.tpg.missing_undesirable_item_index(p.undesirable)
            result[p_index] += pref_value * (ppr[d_index] + ppr[u_index])
            result[d_index] += item_value * ppr[p_index]
            result[u_index] += item_value * ppr[p_index]
        info('\b' * 38, show)  # delete percentage

        return result

    def personalized_vector(self, user):
        """return an empty vector with only a one in the cell corresponding to
           the user (since it is sparse, let us save memory with a Compressed
           Sparse Row format)
        """
        pv = dok_matrix((self.tpg.number_of_nodes, 1))
        pv[self.tpg.users[user]['index'], 0] = 1
        return pv.tocsc()

    def gr(self, item, rank_vector, min_prob=float_info.min, max_prob=1):
        """return the GRank score given to item by Personalized Page Rank"""
        # probability that random walker pass from desirable item face
        PPR_id = rank_vector[self.tpg.missing_desirable_item_index(item), 0]
        # probability that random walker pass from undesirable item face
        PPR_iu = rank_vector[self.tpg.missing_undesirable_item_index(item), 0]
        if any((PPR_id < min_prob, PPR_id > max_prob, PPR_iu < min_prob,
                PPR_iu > max_prob)):
            # since these are probabilities they should be in (0, 1]
            raise SystemExit(f'\nWARNING: GR({item}) is either 0, 1 or NaN; '
                             f'which are all values not allowed!\n\nPlease '
                             f'reduce the convergence threshold in order to '
                             f'get better recommendations :)\n')
        return PPR_id / (PPR_id + PPR_iu)

    def run_recommendation_algorithm(self, user, show):
        """creates list of tuples (<str item_id>, <float GR(item)>)"""

        # initialize PPR_t=0 randomly but in a way which gives reproducible
        # results, aka the seed of the random generator is initialized with
        # some information user-dependant
        PPR = RandomState(seed=abs(hash(user)) % 2**32).rand(
            self.tpg.number_of_nodes, 1)
        PPR = PPR / PPR.sum()  # normalize PPR

        # since this is constant for the whole algorithm let us compute it
        # outside the convergence loop
        one_minus_alpha_dot_pv = \
            (1 - self.alpha) * self.personalized_vector(user)

        # preload T1 (a part of transition matrix T)
        assert isinstance(self.transition_matrix_1, csr_matrix), \
            'GRank.transition_matrix_1 is not instance of ' \
            'scipy.sparse.csr_matrix'

        info(
            f' Computing recommendations for user {user} '.center(80, '=') +
            '\n\n', show)
        for it in range(args.max_iter):
            info(f'(iteration {it + 1:>4}/{args.max_iter:<4})', show)
            PPR_before = PPR
            # since the following one is the most heavy operation in the whole
            # script it is done on parallel with threads
            PPR = self.alpha_dot_transition_matrix_dot(PPR, show) \
                + one_minus_alpha_dot_pv
            # compute the difference between two iterations
            delta_PPR = norm(PPR - PPR_before)
            info(f'{" " * 4}norm(PPR(t) - PPR(t-1)):{delta_PPR:>15.9f}\n',
                 show)
            # and stop convergence if the norm of the difference is lower than
            # the given threshold
            if delta_PPR < args.threshold:
                break
        else:
            # print a warning if maximum number of iterations is exceeded
            info(
                'WARNING: Maximum number of iterations reached '
                f'({args.max_iter}) stop forced!\n', show)
        info('\n', show)  # empty line
        # sort items by decresing value of GR(item)
        ret = sorted([(str(i), self.gr(i, PPR)) for i in self.tpg.items],
                     key=lambda t: t[1],
                     reverse=True)
        self._recommendation_output[user] = ret

    def top_k_recommendations(self, user, k, show=False):
        """return top k item to recommend to user
           (if not yet done it also run the recommendation algorithm for the
            given user)
        """
        if user not in self._recommendation_output:
            # we did not run the recommendation algorithm
            # for this user yet let us do it right now
            self.run_recommendation_algorithm(user, show)
        if args.only_new:
            # user requested recommendations of items that
            # target users have not yet reviewed/bought
            # let us iterate from the top ranked items
            # to the bottom ones, looking for k new/never-seen items
            ret = list()
            for item, gr_item in self._recommendation_output[user]:
                if len(ret) >= k:
                    break  # we already collected k item
                if item not in self.tpg.dataset.training_set:
                    # we are recommending an item from the test set; very good!
                    ret.append((item, gr_item))
                    continue
                for _, users in self.tpg.dataset.training_set[item].items():
                    if user in users:
                        # unfortunately this item has already been reviewed
                        # let us go on to the next one
                        break
                else:
                    # we are recommending an item which the user did not
                    # review/bought yet; good job!
                    ret.append((item, gr_item))
                    continue
        else:
            # user did not request new/never-seen items
            # let us simply return the top k ranked items
            ret = self._recommendation_output[user][:k]
        assert len(ret) == k, f'ERROR: top_k_recommendations() output length' \
                              f' is not k ({k}) as expected but {len(ret)}'
        return tuple(((item, float(gr_item)) for item, gr_item in ret))

    def ndcg(self, user, k, show=False):
        """please look normalized_discounted_cumulative_gain docstring"""
        return self.normalized_discounted_cumulative_gain(user, k, show=show)

    def normalized_discounted_cumulative_gain(self, user, k, show=False):
        """compute accuracy of top k recommended items for the given user"""

        # this is a list of tuples like: (<str item_id>, <float GR(item_id)>)
        recommended_item_list = self.top_k_recommendations(user, k, show)

        # extract from training set and test set
        # the ratings the user gave to recommended items
        rating = dict()
        for item, gr_score in recommended_item_list:
            try:
                for source in (self.tpg.dataset.test_set,
                               self.tpg.dataset.training_set):
                    if item not in source:
                        continue
                    for star, users in source[item].items():
                        if user in users:
                            rating[item] = int(star)
                            # let us use an exception to exit
                            # the two nested for loops
                            raise Exception('item rating found')
            except Exception as e:
                if str(e) == 'item rating found':
                    continue  # let us look for the next item rating
                raise e  # propagate exceptions not intended to exit the loops
            else:
                # if user did not rate this item we estimate a probable
                # rating with the mean of the other ratings this user did
                rating[item] = mean([
                    int(star) for source in (self.tpg.dataset.test_set,
                                             self.tpg.dataset.training_set)
                    for item, d in source.items() for star, users in d.items()
                    if user in users
                ])
        discounted_cumulative_gain = sum(
            (pow(2, rating[item]) - 1) / log2(i + 2)  # because i starts from 0
            for i, (item, _) in enumerate(recommended_item_list))
        ideal_discounted_cumulative_gain = sum(
            (pow(2, 5) - 1) / log2(i + 2)  # because i starts from 0
            for i, (item, _) in enumerate(recommended_item_list))
        ret = discounted_cumulative_gain / ideal_discounted_cumulative_gain
        return float(ret)


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')
assert version_info >= (3, 6), 'Please use at least Python 3.6'

try:
    # use the Dumper from the compiled C library (if present)
    # because it is faster than the one for the python iterpreter
    yaml_dumper = yaml.CDumper
except AttributeError:
    yaml_dumper = yaml.Dumper  # fallback interpreted and slower Dumper
yaml_kwargs = dict(Dumper=yaml_dumper, default_flow_style=False)

# bind signal handler to Ctrl + C signal in order to avoid awful stack
# traces if user stops the script
signal(SIGINT, sigint_handler)

# build command line argument parser
parser = ArgumentParser(description='\n\t'.join(
    ('', 'Apply the collaborative-ranking approach called GRank to a dataset ',
     'of Amazon reviews.', '',
     'Input file should be in one of the following supported formats:',
     f'\t.{", .".join(Dataset.allowed_input_formats)} file.', '',
     'And it should contain a dictionary like:', '\t{"test_set": {',
     '\t\t"<asin>": {"5": <list of reviewerID>,',
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
                    type=open)
parser.add_argument('-s',
                    '--stop-after',
                    default=None,
                    help='stop script after doing recommendations for a '
                    'certain number of users',
                    metavar='int',
                    type=int)
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
parser.add_argument('-i',
                    '--max-iter',
                    default=20,
                    help='stop convergence after max-iter iterations '
                    '(default: 20)',
                    metavar='int',
                    type=int)
parser.add_argument('-t',
                    '--threshold',
                    default=1e-5,
                    help='stop convergence if: '
                    '"norm(PPR_t - PPR_t-1) < threshold" (default: 1e-5)',
                    metavar='float',
                    type=float)
parser.add_argument('-n',
                    '--only-new',
                    action='store_true',
                    default=True,
                    help='force the recommendation of item the user has '
                    'not bought/reviewed in the training set (default: true)')
parser.add_argument(
    '-j',
    '--parallel',
    default=cpu_count(),
    help=f'process n users in parallel (default: {cpu_count()})',
    metavar='int',
    type=int)
args = parser.parse_args()  # parse command line arguments

if args.stop_after is not None and args.stop_after < 1:
    parser.error('-s/--stop-after must be greater than one')
if args.parallel < 1:
    parser.error('-j/--parallel must be at least 1')
if args.max_iter < 1:
    parser.error('-i/--max-iter must be at least 1')

try:
    # ensure all the k are positive integers sorted by decreasing values
    args.top_k = [10] if not args.top_k else [int(k) for k in args.top_k]
    args.top_k = sorted(set([k for k in args.top_k if k > 0]), reverse=True)
except ValueError as e:
    if 'invalid literal for int' in str(e):
        parser.error('-k/--top-k only takes integers values')
    raise SystemExit(f'ERROR: {str(e)}')

grank = GRank(Dataset(args.input))
info(grank.tpg.dataset.specs)
info(grank.tpg.specs)
info(grank.specs)

# write command line arguments to output file (improves reproducibility)
output_header = ' Command Line Arguments '.center(80, '#') + '\n'
output_header += yaml.dump(
    {
        'args.input': args.input.name,
        'args.max_iter': args.max_iter,
        'args.only_new': args.only_new,
        'args.output': getattr(args.output, 'name', None),
        'args.stop_after': args.stop_after,
        'args.parallel': args.parallel,
        'args.threshold': args.threshold,
        'args.top_k': args.top_k,
    }, **yaml_kwargs)
output_header += ' Results '.center(80, '#') + '\n'
if args.output is not None:
    args.output.write(output_header)
    args.output.flush()
    if args.output != stdout:
        args.output.close()  # close output file if != stdout

stderr_buff, stderr_lock = '', Lock()  # create a buffer and a lock for stderr

# create an input and and output queue for the pool of background workers
job_queue, ret_queue = Queue(), Queue()
workers = [
    Process(target=process_body, args=(job_queue, ret_queue))
    for _ in range(args.parallel)
]

for p in workers:
    p.start()  # spawn workers in background

# fill workers' input queue with the desired number of users to explore
for i, target_user in enumerate(sorted(grank.tpg.users)):
    if args.stop_after is None or i < args.stop_after:
        job_queue.put(target_user)

# fill workers' input queue with no-more-work-messages
for p in workers:
    job_queue.put(None)

results = dict()  # let us collect worker results in this dictionary
for _ in range(args.parallel):  # loop until all workers finish their job
    while True:
        user, k, top_k_ranking, ndcg = ret_queue.get()
        if any((user is None, k is None, top_k_ranking is None, ndcg is None)):
            break  # a worker finished its job, let us exit while-true-loop

        if user not in results:
            results[user] = dict(ranking=dict(), ndcg=dict())

        # fill stderr buffer with recommendations for the current user
        stderr_buff += f' Recommended items for user {user} '.center(80, '=')
        stderr_buff += '\n\n'
        for position, (item, gr) in enumerate(top_k_ranking):
            stderr_buff += f'{position:>4d}) GR(<item {item}>) {gr:>32.6f}\n'
            if position not in results[user]['ranking']:
                results[user]['ranking'][position] = dict(item=item, gr=gr)
        stderr_buff += f'\n{" " * 6}NDCG@{k}'.ljust(15) + f'{ndcg:>46.6f}\n\n'

        results[user]['ndcg'][k] = ndcg

        # rewrite output file with also this user results
        if args.output is not None and args.output != stdout:
            with open(args.output.name, 'w') as f:
                f.write(output_header + yaml.dump(results, **yaml_kwargs))

        # if anybody has the lock or the buffer has too many lines
        if stderr_lock.acquire(block=len(stderr_buff.splitlines()) >= 25):
            info(stderr_buff)  # let us flush the stderr buffer
            stderr_buff = ''
            stderr_lock.release()

for p in workers:
    p.join()  # wait for all workers

# flush any information left in the buffer
stderr_lock.acquire()
info(stderr_buff)
stderr_lock.release()

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
                yaml.dump(ndcg_mean, **yaml_kwargs))

# and write it also on stdout
info(' Mean NDCG@K '.center(80, '=') + '\n\n')
for k in reversed(args.top_k):
    info(f'{" " *6}NDCG@{k}'.ljust(15) + f'{ndcg_mean[k]:>46.6f}\n')
info('\n')
