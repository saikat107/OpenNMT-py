import pickle

import argparse

from codit.clone_based_model import clone_based_structural_transformation
from codit.codit_options_parser import get_options
from codit.grammar import JavaGrammar
from translate_structure import translate_all as structure_translate
import os
from translate_token import main as token_translate
from util import debug


def transform_structurally(structure_options):
    f = open(structure_options.grammar, 'rb')
    # debug('Loading the Grammar')
    grammar = pickle.load(f)
    # debug('Grammar Loaded From : %s' % opt_structure.grammar)
    assert isinstance(grammar, JavaGrammar)
    _, _, all_trees = structure_translate(structure_options, grammar, structure_options.n_best)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    with open(structure_options.tmp_file, 'w') as tmp:
        for trees in all_trees:
            # debug(trees)
            t_strs = [' '.join(tree) for tree in trees]
            wstr = '\t'.join(t_strs)
            tmp.write(wstr + '\n')
        tmp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_structure', '-ms', help='Model For Rule Transformation', required=True)
    parser.add_argument('--model_token', '-mt', help='Model for Token', required=True)
    parser.add_argument('--src_token', '-st', help='Source version file(tokens)', required=True)
    parser.add_argument('--tgt_token', '-tt', help='Target version file(tokens)', required=True)
    parser.add_argument('--src_struct', '-ss', help='Source version file(rules)', required=True)
    parser.add_argument('--beam_size', '-bs', help='Beam Size', default=10)
    parser.add_argument('--n_best', '-nb', help='best K hypothesis', default=1)
    parser.add_argument('--name', '-n', help='Name of the experiment', default='Test')
    parser.add_argument('--grammar', '-g', help='Path of the Grammar file', required=True)
    parser.add_argument('--rule_gen', '-rg', help='Use of Rule generation mechanism',
                        choices=['clone', 'nmt', 'none'], default='none')
    parser.add_argument('--train_rule_src', '-tr_src', help='Path of train rule src file for clone based detection', default=None)
    parser.add_argument('--train_rule_tgt', '-tr_tgt', help='Path of train rule src file for clone based detection', default=None)
    options = parser.parse_args()

    structure_options, token_options = get_options(options)
    if options.rule_gen == 'nmt':
        transform_structurally(structure_options)
    elif options.rule_gen == 'clone':
        assert (options.train_rule_src is not None) and (options.train_rule_tgt is not None), \
            'Train Src and Tgt rules must be provided for clone based structural transformation'
        clone_based_structural_transformation(
            options.train_rule_src, options.train_rule_tgt,
            options.src_struct, 100, options.grammar, 'tmp/generated_node_types_clone_5.nt')
    token_translate(token_options)

    # print(create_tree_from_candidates(['2018 688 1624 1913 1606 469'], grammar))
