import pickle

import argparse

import onmt
from codit.grammar import JavaGrammar
from translate_structure import translate_all as structure_translate
import os, sys

from util import debug


def transform_structurally(opt_structure):
    opt_structure.beam_size *= 10
    actual_n_best = opt_structure.n_best
    opt_structure.n_best *= 10
    f = open(opt_structure.grammar, 'rb')
    debug('Loading the Grammar')
    grammar = pickle.load(f)
    debug('Grammar Loaded From : %s' % opt_structure.grammar)
    assert isinstance(grammar, JavaGrammar)
    _, _, all_trees = structure_translate(opt_structure, grammar, actual_n_best)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    with open(opt_structure.tmp_file, 'w') as tmp:
        for trees in all_trees:
            t_strs = [' '.join(tree) for tree in trees]
            wstr = '\t'.join(t_strs)
            tmp.write(wstr + '\n')
        tmp.close()


def get_structure_transformation_parser():
    parser_structure = argparse.ArgumentParser(
        description='full_translation.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser_structure)
    onmt.opts.translate_opts(parser_structure)
    # parser_structure.add_argument('--name', help='Name of the Experiment', required=True)
    parser_structure.add_argument('--grammar', help='Path of the grammar file', required=True)
    parser_structure.add_argument('-tmp_file', default='tmp/generated_node_types.nt')
    return parser_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_structure', help='Model For Rule Transformation', required=True)
    parser.add_argument('-model_token', help='Model for Token', required=True)
    parser.add_argument('-src_token', help='Source version file(tokens)', required=True)
    parser.add_argument('-src_struct', help='Source version file(rules)', required=True)
    parser.add_argument('-beam_size', help='Beam Size', default=10)
    parser.add_argument('-n_best', help='best K hypothesis', default=1)
    parser.add_argument('-name', help='Name of the experiment', default='Test')
    parser.add_argument('-grammar', help='Path of the Grammar file', required=True)
    options = parser.parse_args()

    opt_structure = get_structure_transformation_parser().parse_args(
        (' -src ' + options.src_struct
         + ' -model ' + options.model_structure + ' -beam_size 100 -n_best 100 -gpu 0 '
         + ' --grammar ' + options.grammar).split())
    transform_structurally(opt_structure)

    # print(create_tree_from_candidates(['2018 688 1624 1913 1606 469'], grammar))
