import pickle

import argparse
import sys, os
from codit.clone_based_model import clone_based_structural_transformation
from codit.codit_options_parser import get_options
from codit.grammar import JavaGrammar
from translate_structure import translate_all as structure_translate
import os
from translate_token import main as token_translate
from util import debug


def transform_structurally(structure_opts):
    if os.path.exists(structure_opts.tmp_file):
        debug('Structure Transformation result already exists!\n')
        return
    f = open(structure_opts.grammar, 'rb')
    debug('Loading the Grammar')
    grammar = pickle.load(f)
    debug('Grammar Loaded From : %s' % structure_opts.grammar)
    assert isinstance(grammar, JavaGrammar)
    _, _, all_trees = structure_translate(structure_opts, grammar, structure_opts.n_best)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    with open(structure_opts.tmp_file, 'w') as tmp:
        for trees in all_trees:
            # debug(trees)
            t_strs = [' '.join(tree) for tree in trees]
            wstr = '\t'.join(t_strs)
            tmp.write(wstr + '\n')
        tmp.close()


def get_paths(dataset_str):
    dataset_dir = {
        'icse': '/home/saikatc/Research/OpenNMT-py/rule_based_data/raw',
        'codit': '/home/saikatc/Research/OpenNMT-py/c_data/raw'
    }
    model_dir = {
        'icse': '/home/saikatc/Research/OpenNMT-py/rule_based_models',
        'codit': '/home/saikatc/Research/OpenNMT-py/c_models'
    }
    model_prefix = {
        'icse': {
            'all': 'all',
            'filtered': 'filtered'
        },
        'codit': {
            'all': 'original',
            'filtered': 'filtered'
        }
    }
    parts = dataset.split('-')
    _data = parts[0]
    _kind = parts[1]  # all, filtered
    _type = parts[2]  # concrete abstract original
    _type_m = parts[2]
    if len(parts) > 3:
        _type += ('_' + parts[3])
        _type_m += ('.' + parts[3])
    _data_path = dataset_dir[_data] + '/' + _kind + '/' + _type
    _model_base = model_dir[_data] + '/' + model_prefix[_data][_kind] + '.' + _type_m + '.'
    tree_count = '1'
    if len(parts) > 4:
        tree_count = parts[4].strip()
    return _data_path, _model_base, tree_count


if __name__ == '__main__':
    dataset = sys.argv[1]
    data_path, model_base, tree_count = get_paths(dataset)
    augmented_token_model = model_base + 'augmented.token-best-acc.pt'
    structure_model = model_base + 'rule-best-acc.pt'
    src_token = data_path + '/test_new/prev.augmented.token'
    tgt_token = data_path + '/test_new/next.augmented.token'
    src_struc = data_path + '/test_new/prev.rule'
    grammar = data_path + '/grammar.bin'
    tmp_file = dataset
    name = dataset
    atc_file_path = data_path + '/test_new/atc_scope.bin'
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_structure', '-ms', help='Model For Rule Transformation',
                        default=structure_model)
    parser.add_argument('--model_token', '-mt', help='Model for Token',
                        default=augmented_token_model)
    parser.add_argument('--src_token', '-st', help='Source version file(tokens)',
                        default=src_token)
    parser.add_argument('--tgt_token', '-tt', help='Target version file(tokens)',
                        default=tgt_token)
    parser.add_argument('--src_struct', '-ss', help='Source version file(rules)',
                        default=src_struc)
    parser.add_argument('--beam_size', '-bs', help='Beam Size', default=50)
    parser.add_argument('--n_best', '-nb', help='best K hypothesis', default=10)
    parser.add_argument('--name', '-n', help='Name of the experiment',
                        default=name)
    parser.add_argument('--grammar', '-g', help='Path of the Grammar file',
                        default=grammar)
    parser.add_argument('--rule_gen', '-rg', help='Use of Rule generation mechanism',
                        choices=['clone', 'nmt', 'none'],
                        default='nmt')
    parser.add_argument('--train_rule_src', '-tr_src', help='Path of train rule '
                                                            'src file for clone based detection', default=None)
    parser.add_argument('--train_rule_tgt', '-tr_tgt', help='Path of train rule '
                                                            'src file for clone based detection', default=None)
    parser.add_argument('-cout',
                        default=tmp_file)
    parser.add_argument('--tree_count', default=tree_count)
    parser.add_argument('--atc', default=atc_file_path)
    options = parser.parse_args('')
    options.name = options.name + '_' + str(options.n_best)
    structure_options, token_options = get_options(options)
    if options.rule_gen == 'nmt':
        transform_structurally(structure_options)
    elif options.rule_gen == 'clone':
        assert (options.train_rule_src is not None) and (options.train_rule_tgt is not None), \
            'Train Src and Tgt rules must be provided for clone based structural transformation'
        clone_based_structural_transformation(
            options.train_rule_src, options.train_rule_tgt,
            options.src_struct, 100, options.grammar, 'tmp/' + options.cout)

    token_translate(token_options)

    # print(create_tree_from_candidates(['2018 688 1624 1913 1606 469'], grammar))
