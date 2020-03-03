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

from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter, \
    SUPPRESS
from math import ceil, floor
from matplotlib import colors, pyplot as plt
from matplotlib.cm import nipy_spectral
from matplotlib.legend_handler import HandlerTuple
from numpy import array, linspace, power, quantile, seterr, zeros
from os import environ
from sklearn.metrics import confusion_matrix
from sys import stderr, version_info
import json
import matplotlib.lines as mlines
import yaml


def debug(msg='', min_level=1, max_level=99):
    """print messages on stderr"""
    if args.loglevel and args.loglevel in range(min_level, max_level + 1):
        print(msg, file=stderr)


def accuracy(contingency_table):
    """(true positives + true negatives) / total cases

       actually sklearn returns a contingency_table like:
        true_negatives | false_positives
       ----------------+----------------
       false_negatives | true_positives
    """
    old_settings = seterr(invalid='raise')
    try:
        return float(contingency_table[1, 1] +
                     contingency_table[0, 0]) / contingency_table.sum()
    except FloatingPointError as e:
        if float(contingency_table.sum()) < 1:
            debug('division by zero avoided, return Nan', min_level=2)
        else:
            debug(str(e), min_level=2)
        return float('nan')
    finally:
        seterr(**old_settings)


def f_score(contingency_table, beta=1):
    """return the harmonic average of precision and recall (with beta=1)"""
    return ((1 + power(beta, 2)) * precision(contingency_table) *
            recall(contingency_table)) / (
                (power(beta, 2) * precision(contingency_table)) +
                recall(contingency_table))


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


def get_top_quartile_and_real_ratings(user, items, dataset):
    rating_list = list()
    rating_dict = dict()
    for source in (dataset['test_set'], dataset['training_set']):
        for item in source:
            for star, user_list in source[item].items():
                if user in user_list:
                    if item in items:
                        rating_dict[item] = float(star) / 5.0
                    rating_list.append(float(star) / 5.0)
    top_quartile = quantile(a=rating_list, q=args.quantile_threshold)
    return top_quartile, rating_dict


def metrics_plot(ax,
                 method,
                 data,
                 ylabel,
                 xlabel=None,
                 xticks=None,
                 hide_ticks=True):
    dataset, cont_table = zip(*sorted(
        data.items(),
        key=lambda t: tuple(
            map(lambda mt: int(mt.lstrip('MT')), '.'.join(t[0].split('/')[
                -1].split('.')[:-1]).split('_')))))

    x, y, lines = dict(), dict(), dict()
    for dataset_name, metric in zip(dataset, map(method, cont_table)):
        M_label, T_label = '.'.join(
            dataset_name.split('/')[-1].split('.')[:-1]).split('_')
        lines[M_label] = dict(label=M_label, markersize=8)
        lines[M_label]['marker'] = dict(m1000='x', m300='s',
                                        m100='^').get(M_label.lower(), '.')
        lines[M_label]['color'] = dict(m1000='black',
                                       m300='orange',
                                       m100='blue').get(M_label.lower(), None)
        if M_label not in x:
            x[M_label] = list()
        x[M_label].append(int(T_label.lower().strip('t')))
        if M_label not in y:
            y[M_label] = list()
        y[M_label].append(metric)
    for line, kwargs in lines.items():
        ax.plot(x[line], y[line], **kwargs)

    if xticks is not None:
        ax.set_xlim((xticks[0] - 1, xticks[-1] + 1.5))
        ax.set_xticks(sorted(xticks))
        ax.set_xticklabels([f'T{t}' for t in xticks])
    ax.tick_params(axis='x', color='white' if hide_ticks else 'black')

    y1, y2 = ax.get_ylim()
    y1 = int(5.0 * floor(100.0 * y1 / 5.0))
    y2 = int(5.0 * ceil(100.0 * y2 / 5.0))
    ax.set_ylim((y1 / 100.0, y2 / 100.0))
    ax.set_yticks([round(i / 100.0, 2) for i in range(y1, y2 + 10, 10)])
    ax.set_yticklabels([f'{i / 100.0:.2f}' for i in range(y1, y2 + 10, 10)])
    ax.set_yticks([round(i / 100.0, 2) for i in range(y1, y2 + 5, 5)],
                  minor=True)
    ax.tick_params(axis='y', which='both', length=7)

    if xlabel is not None:
        xlabel = 'GRank' if xlabel.lower() == 'grank' else xlabel
        ax.set_xlabel(xlabel, labelpad=10, fontsize=16, fontweight='bold')
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=6, fontsize=12, fontweight='semibold')


def ndcg_bar_plot(ax, data, ylabel, xlabel=None, hide_ticks=True, vlines=[]):
    dataset, ndcg = zip(*sorted(
        data.items(),
        key=lambda t: tuple(
            map(lambda mt: int(mt.lstrip('MT')), '.'.join(t[0].split('/')[
                -1].split('.')[:-1]).split('_')))))
    for rect in ax.bar(
            x=list(range(len(dataset))),
            height=ndcg,
            color=nipy_spectral(
                list(linspace(0.12, 0.43, int(len(dataset) / 2))) +  # blue
                list(linspace(0.69, 0.97, int(len(dataset) / 2))) +  # red
                [0])):  # black
        ax.annotate(
            f'{rect.get_height():.2f}',
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom')

    ax.set_xticks(list(range(len(dataset))))
    ax.set_xticklabels(dataset, rotation=45, horizontalalignment='right')
    ax.set_yticks(list())  # no y ticks
    ax.set_yticklabels(list())  # no y tick labels
    ax.tick_params(color='white' if hide_ticks else 'black')

    ax.vlines([float(x) - 0.5 for x in vlines],
              ymin=0,
              ymax=1.05,
              linestyle='--',
              linewidth=1)

    if xlabel is not None:
        xlabel = 'GRank' if xlabel.lower() == 'grank' else xlabel
        ax.set_xlabel(xlabel, labelpad=0, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, labelpad=6, fontsize=12, fontweight='semibold')
    ax.set_ylim(0, 1.05)


def time_plot(ax,
              data,
              xlabel,
              ylabel=None,
              xticks=None,
              hide_ticks=True,
              show_parallel_time=False):
    dataset, time_table = zip(*sorted(
        data.items(),
        key=lambda t: tuple(
            map(lambda mt: int(mt.lstrip('MT')), '.'.join(t[0].split('/')[
                -1].split('.')[:-1]).split('_')))))
    for key, alpha, style in (('cpu_time', 1, '-'), ('wall_clock_time', 0.4,
                                                     ':')):
        if not show_parallel_time and key == 'wall_clock_time':
            continue
        x, y, lines = dict(), dict(), dict()
        for dataset_name, time_dict in zip(dataset, time_table):
            M_label, T_label = '.'.join(
                dataset_name.split('/')[-1].split('.')[:-1]).split('_')
            lines[M_label] = dict(label=M_label, markersize=8)
            lines[M_label]['marker'] = dict(m1000='x', m300='s', m100='^').get(
                M_label.lower(), '.')
            lines[M_label]['color'] = dict(m1000='black',
                                           m300='orange',
                                           m100='blue').get(
                                               M_label.lower(), None)
            if M_label not in x:
                x[M_label] = list()
            x[M_label].append(int(T_label.lower().strip('t')))
            if M_label not in y:
                y[M_label] = list()
            y[M_label].append(time_dict[key] / 60.0)  # from seconds to minutes
        for line, kwargs in lines.items():
            ax.plot(x[line], y[line], alpha=alpha, linestyle=style, **kwargs)
        if key == 'wall_clock_time':
            ax.legend([
                tuple([
                    mlines.Line2D([], [],
                                  alpha=0.4,
                                  color='black',
                                  linestyle=':',
                                  marker=m) for m in ('x', 's', '^')
                ])
            ], ['run on parallel (8 core)'],
                      numpoints=1,
                      handlelength=4,
                      handler_map={tuple: HandlerTuple(ndivide=None)},
                      loc='upper right')

    if xticks is not None:
        ax.set_xlim((xticks[0] - 1, xticks[-1] + 1.5))
        ax.set_xticks(sorted(xticks))
        ax.set_xticklabels([f'T{t}' for t in xticks])
    ax.tick_params(axis='x', color='white' if hide_ticks else 'black')

    y_up_limit = ax.get_ylim()[1]
    long_times = dict(year=60 * 24 * 365,
                      month=60 * 24 * (365 / 12),
                      two_weeks=60 * 24 * 14,
                      week=60 * 24 * 7,
                      day=60 * 24,
                      hour=60)

    yticks = dict()
    for lab, y in sorted(long_times.items(), key=lambda t: t[1], reverse=True):
        if y_up_limit * 4 / 3 > y and len(yticks) < 4:
            yticks[lab.replace("two_", "2 ")] = y

    if xlabel is not None:
        xlabel = 'GRank' if xlabel.lower() == 'grank' else xlabel
        ax.set_xlabel(xlabel,
                      labelpad=10 / 2 if hide_ticks else 10,
                      fontsize=16,
                      fontweight='bold')
    if ylabel is not None:
        ax.set_ylabel(f'{ylabel}{" (min)" if len(yticks) < 2 else ""}',
                      labelpad=0 if hide_ticks else 10,
                      fontsize=12,
                      fontweight='semibold')

    if len(yticks) < 2:
        ax.axhline(min(yticks.values()),
                   color='black',
                   linestyle='-.',
                   linewidth=1,
                   label=None)
        ax.set_yticklabels([
            f'{round(t):.0f}'
            if t != min(yticks.values()) else min(yticks.keys())
            for t in ax.get_yticks()
        ])
    else:
        ax.set_yticks(sorted(yticks.values()))
        ax.axhline(sorted(yticks.values())[round(len(yticks) / 2)],
                   color='black',
                   linestyle='-.',
                   linewidth=1,
                   label=None)
        ax.set_yticklabels(
            [k for k, v in sorted(yticks.items(), key=lambda t: t[1])])
        ax.set_ylim((min(yticks.values()) * -1.1, max(yticks.values()) * 1.1))
    ax.tick_params(axis='y', which='both', length=7)


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

if str(environ.get('PYTHONHASHSEED', 'unset')) != '0':
    raise SystemExit('Please set environment variable PYTHONHASHSEED '
                     'to 0 (zero) to have reproducible results')

try:
    # use the Loader from the compiled C library (if present)
    # because it is faster than the one for the python iterpreter
    yaml_loader = yaml.CLoader
except AttributeError:
    yaml_loader = yaml.Loader  # fallback interpreted and slower Loader

parser = ArgumentParser(description='\n'.join(
    ('Plot:', '  - NDCG@k', '  - execution time performances',
     '  - accuracy/precision/recall/f-score performances',
     'of two recommendations approaces (GRank and Naïve Bayes)')),
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument(
    '-v',
    '--verbose',
    action='count',
    default=0,
    dest='loglevel',
    help='print verbose messages (multiple -v increase verbosty)\n')
parser.add_argument(dest='input_files',
                    help='yaml results files',
                    metavar='results.yaml',
                    nargs='+',
                    type=FileType('r'))
parser.add_argument('--accuracy',
                    const=False,
                    default=None,
                    dest='accuracy_plot',
                    help='produce accuracy/precision/recall/f-score '
                    'plot interactively or on file',
                    metavar='plot.{pdf,png,svg}',
                    nargs='?',
                    type=FileType('wb'))
parser.add_argument('--confusion-matrix',
                    default=list(),
                    dest='conf_matrix',
                    help='yaml confusion matrix files',
                    metavar='matrix.yaml',
                    nargs='+',
                    type=FileType('r'))
parser.add_argument('--dataset',
                    default=list(),
                    help='json dataset files',
                    metavar='dataset.json',
                    nargs='+',
                    type=FileType('r'))
parser.add_argument('--log-files',
                    default=list(),
                    help='/usr/bin/time -v stderr logging output',
                    metavar='stderr.log',
                    nargs='+',
                    type=FileType('r'))
parser.add_argument('--ndcg',
                    const=False,
                    default=None,
                    dest='ndcg_plot',
                    help='produce ndcg plot interactively or on file',
                    metavar='plot.{pdf,png,svg}',
                    nargs='?',
                    type=FileType('wb'))
parser.add_argument(
    '--time',
    const=False,
    default=None,
    dest='time_plot',
    help='produce execution-time plot interactively or on file',
    metavar='plot.{pdf,png,svg}',
    nargs='?',
    type=FileType('wb'))
parser.add_argument('--width', default=11.7, help=SUPPRESS, type=float)
parser.add_argument('--height', default=8.3, help=SUPPRESS, type=float)
parser.add_argument('--vline', default=list(), help=SUPPRESS, action='append')
parser.add_argument('--quantile',
                    default=0.75,
                    dest='quantile_threshold',
                    help=SUPPRESS,
                    type=float)
args = parser.parse_args()

if args.width < 0 or args.height < 0:
    parser.error('--width and --height must be positive floats')
if args.quantile_threshold > 1 or args.quantile_threshold < 0:
    parser.error('--quantile must be a float in [0, 1]')

suggestions = dict(grank=dict(), naïve_bayes=dict())
contingency_table = dict(grank=dict(), naïve_bayes=dict())
data, datasets = dict(grank=dict(), naïve_bayes=dict()), list()

for f in args.input_files:
    cli_args_dict, suggest_dict, ndgc_dict = parse_yaml_file(f)
    if all(
        ('args.max_iter' in cli_args_dict, 'args.threshold' in cli_args_dict)):
        algorithm = 'grank'
    elif all(('args.scaling_factor' in cli_args_dict,
              'args.test_set_size' in cli_args_dict)):
        algorithm = 'naïve_bayes'
    else:
        parser.error('Could not determine recommendation approach of '
                     f'file "{f.name}", please use only GRank and '
                     'Naive Bayes yaml output files.')
    datasets.append(cli_args_dict['args.input'])
    for k, ndcg in ndgc_dict.items():
        if k not in data[algorithm]:
            data[algorithm][k] = dict()
        data[algorithm][k][cli_args_dict['args.input']] = ndcg
    if cli_args_dict['args.input'] not in suggestions[algorithm]:
        suggestions[algorithm][cli_args_dict['args.input']] = suggest_dict

all_k = sorted(set({k for d in data.values() for k in d.keys()}), reverse=True)
for d in datasets:
    for algorithm in data.keys():
        for k in all_k:
            if d not in data[algorithm][k]:
                data[algorithm][k][d] = -1  # fill missing values

wait = False
if args.ndcg_plot is not None:
    fig, axes = plt.subplots(nrows=len(all_k),
                             ncols=len(data.keys()),
                             sharex='col',
                             figsize=(args.width, args.height))
    plt.subplots_adjust(left=0.07,
                        bottom=0.21,
                        right=0.997,
                        top=0.998,
                        wspace=0.1,
                        hspace=0.2)

    for k, ax_row in zip(all_k, axes):
        for algorithm, ax in zip(data.keys(), ax_row):
            ndcg_bar_plot(ax,
                          data[algorithm][k],
                          xlabel=' '.join(
                              (a.capitalize() for a in algorithm.split('_')))
                          if k == min(all_k) else None,
                          ylabel=f'NDCG @ {k}',
                          hide_ticks=k != min(all_k),
                          vlines=args.vline)

    if not args.ndcg_plot:
        plt.show(block=False)
        wait = True
    else:
        fig.savefig(args.ndcg_plot.name)
        plt.close(fig)
if args.accuracy_plot is not None:
    for dataset_file, results in suggestions['grank'].items():
        for file_object in args.dataset:
            if dataset_file == file_object.name:
                dataset = json.load(open(file_object.name))
                break
        else:
            parser.error(f'Could not find {dataset_file} in --dataset values')
        contingency_table['grank'][dataset_file] = zeros((2, 2))
        for user in results:
            items = [p['item'] for p in results[user]['ranking'].values()]
            top_quartile, rating_dict = get_top_quartile_and_real_ratings(
                user, items, dataset)
            if not rating_dict:
                continue
            y_real = array(list(rating_dict.values())) >= top_quartile
            y_predicted = array([
                p['rating'] for p in results[user]['ranking'].values()
                if p['item'] in rating_dict.keys()
            ]) >= top_quartile

            contingency_table['grank'][dataset_file] += confusion_matrix(
                y_real, y_predicted)

    for g in args.conf_matrix:
        d = yaml.load(g, Loader=yaml_loader)
        if d['args.input'] in datasets:
            contingency_table['naïve_bayes'][d['args.input']] = array(
                d['confusion_matrix'])

    for g in args.dataset:
        for algorithm in contingency_table.keys():
            if g.name not in contingency_table[algorithm]:
                debug(f'WARNING: contingency table of {algorithm} '
                      f'on {g.name} not found!')
                contingency_table[algorithm][g.name] = zeros((2, 2))

    fig2, axes2 = plt.subplots(nrows=4,
                               ncols=2,
                               sharex='col',
                               figsize=(2.0 * args.width / 3.0, args.height))

    xticks = sorted(
        set(
            int('.'.join(fn.split('/')[-1].split('.')[:-1]).split('_')
                [1].lower().strip('t')) for d in contingency_table.values()
            for fn in d.keys()))

    for ax_row, metric, metric_name in zip(
            axes2, (accuracy, precision, recall, f_score),
        ('Accuracy', 'Precision', 'Recall', 'F-score')):
        for algorithm, ax in zip(contingency_table.keys(), ax_row):
            metrics_plot(
                ax,
                metric,
                contingency_table[algorithm],
                xlabel=' '.join((a.capitalize() for a in algorithm.split('_')))
                if metric_name == 'F-score' else None,
                ylabel=metric_name if algorithm == 'grank' else None,
                xticks=xticks,
                hide_ticks=metric_name != 'F-score',
            )

    for ax_row in axes2:
        y1 = min(ax.get_ylim()[0] for ax in ax_row)  # new_min
        y2 = max(ax.get_ylim()[1] for ax in ax_row)  # new_max
        y1 = int(5.0 * floor(100.0 * y1 / 5.0))
        y2 = int(5.0 * ceil(100.0 * y2 / 5.0))
        for ax in ax_row:
            ax.set_ylim((y1 / 100.0, y2 / 100.0))
            ax.set_yticks(
                [round(i / 100.0, 2) for i in range(y1, y2 + 10, 10)])
            ax.set_yticklabels(
                [f'{i / 100.0:.2f}' for i in range(y1, y2 + 10, 10)])
            ax.set_yticks([round(i / 100.0, 2) for i in range(y1, y2 + 5, 5)],
                          minor=True)
            ax.tick_params(axis='y', which='both', length=7)

    handles, labels = list(), list()
    for ax_row in axes2:
        for ax in ax_row:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    labels.append(label)
                    handles.append(handle)

    fig2.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        mode='expand',
        bbox_to_anchor=(0.08, 0.890, 0.925, 0.1))
    plt.subplots_adjust(
        left=0.09,
        bottom=0.07,
        right=0.997,
        top=0.935,
        wspace=0.20,
        hspace=0.10)

    if not args.accuracy_plot:
        plt.show(block=False)
        wait = True
    else:
        fig2.savefig(args.accuracy_plot.name)
        plt.close(fig2)
if args.time_plot is not None:
    execution_times = dict(grank=dict(), naïve_bayes=dict())
    for f in args.log_files:
        text = f.read()
        cpus = 1
        seconds = 0
        wall_clock_secs = 0
        for line in text.lower().split('\n'):
            if 'percent of cpu' in line and '%' in line:
                cpus = float(line.split(':')[-1].strip(' %')) / 100.0
            elif 'user time' in line and 'seconds' in line:
                seconds += float(line.split(':')[-1].strip(' '))
            elif 'system time' in line and 'seconds' in line:
                seconds += float(line.split(':')[-1].strip(' '))
            elif 'wall clock' in line and 'h:mm:ss' in line:
                time = line.split(': ')[-1].strip(' ')
                if '.' in time:
                    hours = 0
                    mins, secs = map(float, time.split(':'))
                else:
                    hours, mins, secs = map(float, time.split(':'))
                wall_clock_secs = secs + 60.0 * mins + 60.0 * 60.0 * hours
            else:
                continue
        for file_object in args.dataset:
            if file_object.name in text:
                algorithm = 'grank' if 'grank' in text else 'naïve_bayes'
                execution_times[algorithm][file_object.name] = dict(
                    cpus=cpus,
                    cpu_time=seconds,
                    wall_clock_time=wall_clock_secs)
                break
        else:
            parser.error(f'Could not find dataset of logfile {f.name} '
                         'in --dataset values')

    xticks = sorted(
        set(
            int('.'.join(fn.split('/')[-1].split('.')[:-1]).split('_')
                [1].lower().strip('t')) for d in execution_times.values()
            for fn in d.keys()))

    fig3, axes3 = plt.subplots(nrows=2,
                               ncols=1,
                               sharex='col',
                               figsize=(1.0 * args.width / 3.0, args.height))

    for ax, algorithm in zip(axes3, list(data.keys())):
        time_plot(
            ax,
            execution_times[algorithm],
            xlabel=' '.join((a.capitalize() for a in algorithm.split('_'))),
            ylabel='Time',
            xticks=xticks,
            hide_ticks=algorithm == 'grank',
            show_parallel_time=algorithm == 'grank',
        )
    handles, labels = list(), list()
    for ax in axes3:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels and label is not None:
                labels.append(label)
                handles.append(handle)
    fig3.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        mode='expand',
        bbox_to_anchor=(0.17, 0.890, 0.84, 0.1))
    plt.subplots_adjust(
        left=0.19,
        bottom=0.07,
        right=0.993,
        top=0.935,
        wspace=0.17,
        hspace=0.1)
    if not args.time_plot:
        plt.show(block=False)
        wait = True
    else:
        fig3.savefig(args.time_plot.name)
        plt.close(fig3)
if wait:
    plt.show()
