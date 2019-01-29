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

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from math import log2
from numpy import mean, zeros
from numpy.linalg import norm
from numpy.random import RandomState
from queue import Queue
from scipy.sparse import dok_matrix
from signal import pthread_kill, signal, SIGINT, SIGTERM, SIGKILL
from sys import float_info, getsizeof, stderr
from threading import Thread, Lock, current_thread, main_thread
import json
import pickle
import yaml


def info(message='', *args, **kwargs):
    print(message, *args, **kwargs, file=stderr)


def sigint_handler(sig_num, stack_frame):
    """Cleanily catch SIGINT and exit without printing long stack traces"""
    if sig_num != SIGINT.value:
        return
    info(f'\n\nReceived SIGINT (Ctrl + C)')
    for sig in (SIGTERM, SIGKILL):
        info(f'Sending {sig.name} to all threads')
        for t in threads:
            if isinstance(t, Thread) and t.ident is not None:
                pthread_kill(t.ident, sig)
    raise SystemExit()


def worker(ppr_vector, alpha, N):
    """this function is the body of threads spawned during the execution of
       the matrix-vector product (alpha * T2 * PPR).

       Each thread has its queue from which it takes (one by one) the workloads
       it has to process. A workload is a tuple like:
           (preference_index, desirable_item_index, undesirable_item_index)
       which represents the indexes of three nodes of the graph (a preference
       from 2nd layer and its two respective and linked desirable/undesirable
       nodes from 3rd layer).  Betweeen theese three nodes there are two
       undirect edges which are mapped to two different rows of the transition
       matrix and the two respective transposed different columns (because the
       graph is undirect).  With this piece of information (2 rows and 2
       columns) it is possible to execute several independent row-vs-col
       matrix-vector products on parallel.
    """
    global queues, result, result_lock  # global variables for multithreading
    t_id = int(current_thread().name)  # id of this thread
    # local variable with the sum of all the results this thread computed
    local_vector = zeros(ppr_vector.shape)
    # values of T2 non zero cells in ...
    pref_value = float(alpha) / 2.0  # ... block 2
    item_value = float(alpha) / float(N - 1)  # ... block 6
    while True:
        # get a workload from the main thread through a queue
        pref_index, item_d_index, item_u_index = queues[t_id].get()
        if pref_index is None or item_d_index is None or item_u_index is None:
            # that workload tell us that there is nothing more to do,
            # let us sum our local results with the global ones
            result_lock.acquire()
            result += local_vector
            result_lock.release()
            break
        # compute the three row-vs-col matrix-vector products corresponding to
        # the workload
        local_vector[pref_index] += pref_value * \
            (ppr_vector[item_d_index, 0] + ppr_vector[item_u_index, 0])
        local_vector[item_d_index] += item_value * ppr_vector[pref_index, 0]
        local_vector[item_u_index] += item_value * ppr_vector[pref_index, 0]
        queues[t_id].task_done()  # tell the queue we executed this workload


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
        return all((self.user == other.user,
                    self.preference == other.preference))

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
        return '\n'.join((
            f'',
            f'Tripartite Graph specs:',
            f'n° edges: {self.number_of_edges: >21}',
            f'n° nodes: {self.number_of_nodes: >21}',
            f'├── user layer: {self.M: >15}{both}',
            f'├── preference layer: {self.N * (self.N - 1): >9}{train}',
            f'│   (of which only {len(self.preferences): >12}',
            f'│    linked to users)',
            f'│',
            f'└── (un)desirable layer: {2 * self.N: >6}{both}',
        ))

    @property
    def users(self):
        """dictionary of <str user_id>: <int user_degree>
           each user_id corresponds to a node in 1st layer
        """
        return self._users

    @property
    def M(self):
        """number of users (nodes in 1st layer)"""
        return self._m

    @property
    def observations(self):
        """set of Observations, each one corresponds to a link between
           1st and 2nd layer
        """
        return self._observations

    @property
    def preferences(self):
        """dictionary of <Preference pref_obj>: <int preference_degree>
           each pref_obj corresponds to a node in 2nd layer"""
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
                    if preference not in self.preferences.keys():
                        yield (preference, index)
                        index += 1

    @property
    def number_of_missing_preferences(self):
        """number of nodes in 2nd layer which are not linked
           to any node in the 1st layser
        """
        return self.N * (self.N - 1) - len(self.preferences)

    @property
    def items(self):
        """set of Item, for each one of these two nodes are generated in 3rd
           layer (one for the item desirable face and one for the undesirable
           one)
        """
        return self._items

    @property
    def N(self):
        """number of items (half of the nodes in 3rd layer)"""
        return self._n

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

    def __init__(self, **kwargs):
        # ensure both training set and test set kwargs were explicitly set
        if any(('test_set_reviews' not in kwargs,
                'training_set_reviews' not in kwargs)):
            raise ValueError('ERROR: please set both training_set_reviews and '
                             'test_set_reviews kwargs in TPG constructor.')

        training_set_reviews = kwargs['training_set_reviews']
        test_set_reviews = kwargs['test_set_reviews']

        # create users nodes (with a null degree) and populate items set
        self._items = set()
        self._users = dict()
        for d in training_set_reviews:
            self._users[d['user']] = dict(degree=0)
            self._items.add(Item(d['item']))
        for d in test_set_reviews:
            self._users[d['user']] = dict(degree=0)
            self._items.add(Item(d['item']))
        self._items = tuple(sorted(self._items))
        # initialize properties which return n° items and n° users
        self._n = len(self.items)
        self._m = len(self.users)

        # create preferences nodes, and populate observation set (which
        # corresponds to the edges between 1st and 2nd layer)
        # warning: only preferences explicitly chosen by at least a user are
        #          considered here; those linked only with the 3rd layer will
        #          be generated next (because they are too many to stay in ram)
        # user or preference node degree are also updated any time a new edge,
        # starting/ending from/to it, is hit
        self._preferences = dict()
        self._observations = set()
        for user in self.users:
            # create a dictionary with a key for each n° stars
            user_reviews = {s: set() for s in range(1, 6)}
            for d in (d for d in training_set_reviews if d['user'] == user):
                # add any reviewed item to the corresponding n° stars key
                user_reviews[int(d['stars'])].add(d['item'])

            # iterate over comparisons between items with more stars against
            # those with less stars (i.e. items with 5 stars vs items with 4,
            # ... 5 stars vs 1 star, 4 stars vs 3 stars, ..., 2 stars vs 1)
            for more_stars, less_stars in self.comparisons:
                for asin_d in user_reviews[more_stars]:
                    for asin_u in user_reviews[less_stars]:
                        preference = Preference(Item(asin_d), Item(asin_u))
                        self._observations.add(Observation(user, preference))
                        if preference not in self.preferences:
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
        """return index of desirable items"""
        if getattr(self, '_missing_desirables', None) is None:
            self._missing_desirables = dict()
            for index, item in enumerate(self.items):
                self._missing_desirables[item] = \
                    index + self.M + self.N * (self.N - 1)
        return self._missing_desirables[item]

    def missing_undesirable_item_index(self, item):
        """return index of undesirable items"""
        return self.missing_desirable_item_index(item) + self.N


class GRank():

    allowed_input_formats = ('json', 'pickle', 'yaml')

    @property
    def specs(self):
        return '\n'.join(('',
                          'GRank specs:',
                          f'max iterations: {args.max_iter: >7d}',
                          f'threshold:{" " * 12}{args.threshold:g}',
                          f'alpha: {self.alpha: >19.2f}',
                          ''))

    @property
    def test_set(self):
        """dictionary of <str item_id>: {<str n° star>: [<str user_id>, ...],
                                          ... }
        """
        return self._test_set

    @property
    def training_set(self):
        """dictionary of <str item_id>: {<str n° star>: [<str user_id>, ...],
                                          ... }
        """
        return self._training_set

    @property
    def descriptions(self):
        """dictionary of <str item_id>: <str item description>"""
        return self._descriptions

    @property
    def dataset_specs(self):
        both = ' (both training and test set)'
        train = ' (only training          set)'
        return '\n'.join((
            f'',
            f'Dataset specs:',
            f'n° users: {len(self.tpg.users): >21}{both}',
            f'n° items: {len(self.tpg.items):21}{both}',
            f'n° reviews: {len(self.reviews(training_set=True)): >19}{train}',
            f'n° observations: {len(self.tpg.observations): >14}{train}'
        ))

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
            info(f'Building sparse transition matrix T:{" " * 21}',
                 end='', flush=True)
            # to build this sparse matrix the usage of a Dictionary Of Keys
            # based one is really handy
            t1 = dok_matrix(
                (self.tpg.number_of_nodes, self.tpg.number_of_nodes))
            total = len(self.tpg.observations)
            # iterate over edges between 1st and 2nd layer
            for i, obs in enumerate(self.tpg.observations):
                if i % (total // 10**4) == 0:
                    info('\b' * 7 + f'{100 * i / total: >6.2f}%', end='',
                         flush=True)
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
            info('\b' * 7 + '100.00%')

            # compute and show some informations
            nnz = self._transition_matrix_1.nnz \
                + self.tpg.number_of_missing_preferences * 2 * 2
            size = self.tpg.number_of_nodes ** 2
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
            info('\n'.join((
                f'n° of non zero T elements: {nnz: >26d}',
                f'n° of all T elements: {size: >31d}',
                f'density of T matrix: {density: >42g}',
                f'size of T in ram: {ram_size: >38.2f} {ram_unit}')))
        return self._transition_matrix_1

    def __init__(self, file_object, alpha=0.85):
        # check that input file format is allowed, then load its data
        extension = file_object.name.split('.')[-1]
        if extension not in self.allowed_input_formats:
            raise SystemExit('ERROR: input file format not supported, please '
                             f'use: .{", .".join(GRank.allowed_input_formats)}'
                             ' file.\n')
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
        try:
            self._test_set = data['test_set']
            self._training_set = data['training_set']
            self._descriptions = data['descriptions']
        except KeyError as e:
            raise SystemExit(f'ERROR: {str(e)}')
        else:
            info(f'Successfully loaded dataset from {file_object.name}')

        self._alpha = alpha
        self._transition_matrix_1 = None
        self._recommendation_output = dict()
        self._test_set_reviews = None
        self._training_set_reviews = None

        # build a tripartite graph from the loaded dataset
        self._tpg = TPG(training_set_reviews=self.reviews(training_set=True),
                        test_set_reviews=self.reviews(test_set=True))

    def reviews(self, test_set=False, training_set=False):
        """cache and return a tuple of reviews; a review is a dict like:
           {'user': <str user_id>, 'item': <str item_id>, 'stars': <int stars>}
        """
        if test_set and self._test_set_reviews is None:
            self._test_set_reviews = tuple(self._reviews(test_set=True))
        elif training_set and self._training_set_reviews is None:
            self._training_set_reviews = tuple(
                self._reviews(training_set=True))
        if test_set:
            return self._test_set_reviews
        elif training_set:
            return self._training_set_reviews
        raise ValueError('ERROR: please set test_set or training_set flag')

    def _reviews(self, test_set=False, training_set=False):
        """private reviews generator; a review is a dict like:
           {'user': <str user_id>, 'item': <str item_id>, 'stars': <int stars>}
        """
        if test_set:
            data = self.test_set
        elif training_set:
            data = self.training_set
        else:
            raise ValueError('ERROR: please set test_set or training_set flag')
        for item, d in data.items():
            for stars, reviewers in d.items():
                if stars not in tuple(str(i) for i in range(1, 6)):
                    raise ValueError(f'Invalid stars: "{repr(stars)}"; '
                                     'string expected.')
                for user in reviewers:
                    yield dict(user=str(user), item=str(item), stars=int(stars))

    def alpha_dot_transition_matrix_dot(self, ppr):
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

           Actually   alpha * T2 * PPR   is also computed on parallel with
           threads in order to distribute its heavy computational cost on
           several cores/processors.

           Keep in mind that cell values are:   cell[i, j] = 1 / degree(i)
           Which means that block 4 has cells with 1/2 as value and block 6 has
           cells with 1/(N -1) as value.  Cells in other blocks have a variable
           degree accordingly to how many users expressed a given preference.
        """
        global queues, result, result_lock, threads
        result_lock.acquire()
        # initialize result as the first term of the sum:  alpha * T1 * PPR
        result = self.transition_matrix_1.dot(ppr)
        result_lock.release()

        # initialize needed data structures for a multithreaded execution
        queues = [Queue() for i in range(args.threads)]
        # each thread will execute function worker(ppr, alpha, N)
        threads = [Thread(target=worker,
                          args=(ppr, self.alpha, self.tpg.N),
                          name=str(i))  # threads are named '0', '1', ...
                   for i in range(args.threads)]
        for t in threads:
            t.start()

        total = self.tpg.number_of_missing_preferences
        # iterate over edges between 2nd and 3rd layer
        # (a python generator is used because they would not fit in ram)
        for i, (p, pref_index) in enumerate(self.tpg.missing_preferences):
            if i % (total // 10**4) == 0 and i / total < 1 + 1e-6:
                if i > 0:
                    info('\b' * 7 + f'{100 * i / total: >6.2f}%',
                         end='', flush=True)
                else:
                    info(f'Multiplying    alpha * T * PPR(t-1):{" " * 21}',
                         end='', flush=True)

            id_index = self.tpg.missing_desirable_item_index(p.desirable)
            iu_index = self.tpg.missing_undesirable_item_index(p.undesirable)
            # put the workload described by the tuple:
            # (preference_index, desirable_item_index, undesirable_item_index)
            # into the queues from which threads get their jobs
            #
            # workloads are balanced in a venetian-blind / round-robin fashion
            queues[i % args.threads].put((pref_index, id_index, iu_index))

            assert i < total, f'WARNING: i > total ({i} > {total}'

        for q in queues:
            q.join()  # wait far all elements in the queues to be processed
            q.put((None, None, None))  # send into each queue a stop signal
        for t in threads:
            t.join()  # wait for all threads to return
        info('\b' * 7 + '100.00%')
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
        if any((PPR_id < min_prob, PPR_id > max_prob,
                PPR_iu < min_prob, PPR_iu > max_prob)):
            # since these are probabilities they should be in (0, 1]
            raise SystemExit(f'\nWARNING: GR({item}) is either 0, 1 or NaN;'
                             ' which are all values not allowed!\n\nPlease'
                             ' reduce the convergence threshold in order to'
                             ' get better recommendations :)\n')
        return PPR_id / (PPR_id + PPR_iu)

    def run_recommendation_algorithm(self, user):
        """return a list of tuples (<str item_id>, <float GR(item)>)"""
        info('\n'.join(('=' * 80,
                        'Started recommendation algorithm:',
                        '',
                        f'target user: {user: >40}',
                        f'n° of threads to use: {args.threads: >31d}',
                        '')))

        # initialize PPR_t=0 randomly but in a way which gives reproducible
        # results, aka the seed of the random generator is initialized with
        # some information user-dependant
        PPR = RandomState(
            seed=abs(hash(user)) % 2**32).rand(self.tpg.number_of_nodes, 1)
        PPR = PPR / PPR.sum()  # normalize PPR

        # since this is constant for the whole algorithm let us compute it
        # outside the convergence loop
        one_minus_alpha_dot_pv = \
            (1 - self.alpha) * self.personalized_vector(user)

        # preload T1 (a part of transition matrix T)
        assert self.transition_matrix_1 is not None

        for it in range(1, args.max_iter):
            info(f'\nIteration: {it: >42}')
            PPR_before = PPR
            # since the following one is the most heavy operation in the whole
            # script it is done on parallel with threads
            PPR = self.alpha_dot_transition_matrix_dot(PPR) \
                + one_minus_alpha_dot_pv
            # compute the difference between two iterations
            delta_PPR = norm(PPR - PPR_before)
            info(f'{" " * 13}norm(PPR(t) - PPR(t-1)):{" " * 15}{delta_PPR:g}')
            # and stop convergence if the norm of the difference is lower than
            # the given threshold
            if delta_PPR < args.threshold:
                break
        else:
            # print a warning if maximum number of iterations is exceeded
            info('\nWARNING: Maximum number of iterations reached'
                 f' ({args.max_iter}); stop forced!')
        # sort items by decresing value of GR(item)
        ret = sorted([(str(i), self.gr(i, PPR))
                      for i in self.tpg.items],
                     key=lambda t: t[1],
                     reverse=True)
        info('\nEnded recommendation algorithm\n' + '=' * 80)
        self._recommendation_output[user] = ret

    def top_k_recommendations(self, user, k, show=False):
        """return top k item to recommend to user
           (if not yet done it also run the recommendation algorithm for the
            given user)
        """
        if user not in self._recommendation_output:
            # we did not run the recommendation algorithm for this user yet
            # let us do it right now
            self.run_recommendation_algorithm(user)
        if args.only_new:
            # user requested recommendations of items that
            # target users have not yet reviewed/bought
            # let us iterate from the top ranked items to the bottom ones,
            # looking for k new/never-seen items
            ret = list()
            for suggested_item, gr_item in self._recommendation_output[user]:
                if len(ret) >= k:
                    break  # we already collected k item
                if suggested_item not in self.training_set:
                    # we are recommending an item from the test set; very good!
                    ret.append((suggested_item, gr_item))
                    continue
                for stars, users in self.training_set[suggested_item].items():
                    if user in users:
                        # unfortunately this item has already been reviewed
                        # let us go on to the next one
                        break
                else:
                    # we are recommending an item which the user did not
                    # review/bought yet; good job!
                    ret.append((suggested_item, gr_item))
                    continue
        else:
            # user did not request new/never-seen items
            # let us simply return the top k ranked items
            ret = self._recommendation_output[user][:k]
        assert len(ret) == k, 'ERROR: top_k_recommendations() output length ' \
                              f'is not k {k} as expected but {len(ret)}'
        if show:
            for i, (item, gr) in enumerate(ret):
                if i == 0:
                    info(f'\nRecommended items for target user: {user}')
                info(f'{i: >2d}) GR(<item {item}>): {gr:.6f}')
            else:
                info()
        return tuple(((item, float(gr_item)) for item, gr_item in ret))

    def ndcg(self, user, k, show=False):
        """please look normalized_discounted_cumulative_gain docstring"""
        return self.normalized_discounted_cumulative_gain(user, k, show=show)

    def normalized_discounted_cumulative_gain(self, user, k, show=False):
        """compute accuracy of top k recommended items for the given user"""

        # this is a list of tuples like: (<str item_id>, <float GR(item_id)>)
        recommended_item_list = self.top_k_recommendations(user, k)

        # extract from training set and test set
        # the ratings the user gave to recommended items
        rating = dict()
        for item, gr_score in recommended_item_list:
            try:
                for source in (self.test_set, self.training_set):
                    if item not in source:
                        continue
                    for star, users in source[item].items():
                        if user in users:
                            rating[item] = int(star)
                            # let us use an exception to exit the two for loops
                            raise Exception('item rating found')
            except Exception as e:
                if str(e) == 'item rating found':
                    continue  # let us look for the next item rating
                raise e  # propagate exceptions not intended to exit the loops
            else:
                # if user did not rate that item we estimate a probable
                # rating with the mean of the other ratings this user did
                rating[item] = mean(
                    [int(star)
                     for source in (self.test_set, self.training_set)
                     for item, d in source.items()
                     for star, users in d.items()
                     if user in users])
        discounted_cumulative_gain = sum(
            (pow(2, rating[item]) - 1) / log2(i + 2)  # because i starts from 0
            for i, (item, _) in enumerate(recommended_item_list))
        ideal_discounted_cumulative_gain = sum(
            (pow(2, 5) - 1) / log2(i + 2)  # because i starts from 0
            for i, (item, _) in enumerate(recommended_item_list))
        ret = discounted_cumulative_gain / ideal_discounted_cumulative_gain
        if show:
            info(f'NDCK@{k}: {ret}')
        return float(ret)


if __name__ != '__main__':
    raise SystemExit('Please run this script, do not import it!')

# global variables mainly used for multithreading
global queues, result, result_lock, threads
queues, result, result_lock, threads = None, None, Lock(), list()

# bind signal handler to Ctrl + C signal in order to avoid awful stack traces
# if user stops the script
if current_thread() == main_thread():
    signal(SIGINT, sigint_handler)

# build command line argument parser
parser = ArgumentParser(
    description='\n\t'.join((
        '',
        'Apply the collaborative-ranking approach called GRank to a dataset ',
        'of Amazon reviews.',
        '',
        'Input file should be in one of the following supported formats:',
        f'\t.{", .".join(GRank.allowed_input_formats)} file.',
        '',
        'And it should contain a dictionary like:',
        '\t{"test_set": {\n',
        '\t\t"<asin>": {"5": <list of reviewerID>,',
        '\t\t           "4": <list of reviewerID>,',
        '\t\t           "3": <list of reviewerID>,',
        '\t\t           "2": <list of reviewerID>,',
        '\t\t           "1": <list of reviewerID>},',
        '\t\t  ...',
        '\t\t},',
        '\t"training_set": {',
        '\t\t"<asin>": {"5": <list of reviewerID>,',
        '\t\t           "4": <list of reviewerID>,',
        '\t\t           "3": <list of reviewerID>,',
        '\t\t           "2": <list of reviewerID>,',
        '\t\t           "1": <list of reviewerID>},',
        '\t\t  ...',
        '\t\t},',
        '\t"descriptions": {"<asin>": "description of the item",',
        '\t                   ...      ...',
        '\t\t}',
        '\t}')),
    formatter_class=RawDescriptionHelpFormatter)
parser.add_argument(help='See the above input file specs.',
                    dest='input',
                    metavar='input_file',
                    type=open)
parser.add_argument('-s', '--stop-after',
                    default=None,
                    help='stop script after doing recommendations for a '
                    'certain number of users',
                    metavar='int',
                    type=int)
parser.add_argument('-k', '--top-k',
                    action='append',
                    dest='top_k',
                    help='compute also NDCG@k',
                    metavar='int')
parser.add_argument('-i', '--max-iter',
                    default=20,
                    help='stop convergence after max-iter iterations '
                    '(default: 20)',
                    metavar='int',
                    type=int)
parser.add_argument('-t', '--threshold',
                    default=1e-5,
                    help='stop convergence if: '
                    '"norm(PPR_t - PPR_t-1) < threshold" (default: 1e-5)',
                    metavar='float',
                    type=float)
parser.add_argument('-n', '--only-new',
                    action='store_true',
                    help='force the recommendation of item the user'
                    'has not bought/reviewed in the training set')
parser.add_argument('-j', '--threads',
                    default=8,
                    help='run n threads in parallel (default: 8)',
                    metavar='int',
                    type=int)
args = parser.parse_args()  # parse command line arguments

if args.stop_after is not None and args.stop_after < 1:
    raise SystemExit('ERROR: -s/--stop-after must be greater than one')

args.threads = 1 if args.threads < 1 else args.threads  # force n° threads >= 1
args.max_iter = 1 if args.max_iter < 1 else args.max_iter  # force max_iter >= 1
try:
    # ensure all the k are positive integers sorted by decreasing values
    args.top_k = [10] if not args.top_k else [int(k) for k in args.top_k]
    args.top_k = sorted(set([k for k in args.top_k if k > 0]), reverse=True)
except ValueError as e:
    if 'invalid literal for int' in str(e):
        raise SystemExit('ERROR: -k/--top-k only takes input values')
    raise SystemExit(f'ERROR: {str(e)}')

grank = GRank(args.input)
info(grank.dataset_specs)
info(grank.tpg.specs)
info(grank.specs)

processed_users = 0
for i, target_user in enumerate(grank.tpg.users):
    if args.stop_after is not None and processed_users >= args.stop_after:
        break
    processed_users += 1
    for j, k in enumerate(args.top_k):
        top_k_recommendations = grank.top_k_recommendations(target_user, k,
                                                            show=bool(j == 0))
        ndcg = grank.ndcg(target_user, k, show=True)
