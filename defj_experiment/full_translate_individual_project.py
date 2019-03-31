import pickle

import argparse
import sys

import onmt
from codit.clone_based_model import clone_based_structural_transformation
from codit.grammar import JavaGrammar
from translate_structure import translate_all as structure_translate
import os
from defj_experiment.translate_token import main as token_translate
from util import debug


def get_structure_transformation_parser():
    parser_structure = argparse.ArgumentParser(
        description='full_translation.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser_structure)
    onmt.opts.translate_opts(parser_structure)
    parser_structure.add_argument('--grammar', help='Path of the grammar file', required=True)
    parser_structure.add_argument('--tmp_file', default='tmp/generated_node_types.nt')
    return parser_structure


def get_token_transformation_parser():
    parser = argparse.ArgumentParser(
        description='translate_token.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    parser.add_argument('--name', help='Name of the Experiment')
    parser.add_argument('--tmp_file', default='')
    parser.add_argument('--grammar', required=True)
    parser.add_argument('--atc', default=None)
    parser.add_argument('--tree_count', type=int, default=2)
    parser.add_argument('--files_file')
    parser.add_argument('--parent_tree')
    parser.add_argument('--child_tree')
    return parser


def get_options(options):

    structure_options = get_structure_transformation_parser().parse_args(
        (' -src ' + options.src_struct + ' -batch_size 16'
         + ' -model ' + options.model_structure
         + ' -beam_size ' + str(options.beam_size) + ' -n_best ' + str(50) + ' -gpu 0 '
         + ' --grammar ' + options.grammar + ' --tmp_file tmp/' + options.cout
         + ' -verbose'
         ).split())
    token_options = get_token_transformation_parser().parse_args(
        ('-gpu 0 -model ' + options.model_token + ' -src ' + options.src_token + ' -tgt ' + options.tgt_token
         + ' --name ' + options.name + '.' + options.tree_count + ' -batch_size 1 ' + ' -beam_size '
         + str(options.beam_size) + ' -n_best ' + str(10) + ' --tmp_file tmp/' + options.cout
         + ' --atc ' + options.atc + ' --grammar ' + options.grammar + ' --tree_count ' + str(50)
         + ' --files_file ' + options.files_file + ' --parent_tree ' + options.parent_tree
         + ' --child_tree ' + options.child_tree
         # + ' -verbose '
         ).split())
    return structure_options, token_options


def transform_structurally(structure_opts):
    if os.path.exists(structure_opts.tmp_file):
        #debug('Structure Transformation result already exists!\n')
        return
    tgt = structure_options.src
    tgt = tgt.replace('prev.rule', 'next.token.id')
    debug(tgt)
    inp = open(tgt)
    golden_rules = [line.strip() for line in inp]
    inp.close()
    f = open(structure_opts.grammar, 'rb')
    debug('Loading the Grammar')
    grammar = pickle.load(f)
    debug('Grammar Loaded From : %s' % structure_opts.grammar)
    assert isinstance(grammar, JavaGrammar)
    all_scores, all_rules, all_trees = structure_translate(structure_opts, grammar, structure_opts.n_best)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    correct_rule_count = 0
    with open(structure_opts.tmp_file, 'w') as tmp:
        for trees, rules, scores, golden_rule in zip(all_trees, all_rules, all_scores, golden_rules):
            # debug(trees, scores)

            node_is_str_list = [' '.join(tree) for tree, score in zip(trees, scores)]
            debug('Length : ', len(trees))
            if golden_rule in node_is_str_list:
                debug('Index : ', node_is_str_list.index(golden_rule))
                correct_rule_count += 1
            else:
                debug('Index : ', -1)
            t_strs = [' '.join(tree) + '/' + str(score) for tree, score in zip(trees, scores)]
            wstr = '\t'.join(t_strs)
            tmp.write(wstr + '\n')
        tmp.close()
    debug(correct_rule_count)
    # exit()


if __name__ == '__main__':

    datatype = sys.argv[1]
    tree_count = sys.argv[2]
    sz = sys.argv[3]
    project = sys.argv[4]
    prefix = "/home/saikatc/Research/OpenNMT-py/defj_experiment/"
    data_path = prefix + "data/raw/" + datatype
    model_base = prefix + "models/" + datatype + "."
    augmented_token_model = model_base + 'augmented.token.' + project + '-best-acc.pt'
    structure_model = model_base + 'rule-best-acc.pt'
    src_token = data_path + '/test/' + project + '/prev.augmented.token'
    tgt_token = data_path + '/test/' + project + '/next.augmented.token'
    src_struc = data_path + '/test/' + project + '/prev.rule'
    grammar = data_path + '/grammar.bin'
    files_file = data_path + '/test/' + project + '/files.txt'
    parent_tree = data_path + '/test/' + project + '/prev.tree'
    child_tree = data_path + '/test/' + project + '/next.tree'
    tmp_file = 'defects4j-' + datatype + '-' + project
    name = 'defects4j-method-' + datatype + '-' + project
    atc_file_path = data_path + '/test/' + project + '/atc_scope.bin'
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
    parser.add_argument('--beam_size', '-bs', help='Beam Size', default=int(sz))
    parser.add_argument('--n_best', '-nb', help='best K hypothesis', default=int(sz))
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
    parser.add_argument('--files_file', default=files_file)
    parser.add_argument('--parent_tree', default=parent_tree)
    parser.add_argument('--child_tree', default=child_tree)
    options = parser.parse_args('')
    options.name = options.name + '_' + str(options.n_best)
    structure_options, token_options = get_options(options)
    # debug(token_options)
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
