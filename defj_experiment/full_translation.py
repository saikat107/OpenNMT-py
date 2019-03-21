import pickle

import argparse
import sys, os
from codit.clone_based_model import clone_based_structural_transformation
from codit.codit_options_parser import get_options
from full_translation import transform_structurally
from onmt.translate.translator import build_translator
import os
from translate_token import main as token_translate,\
    extract_atc_from_grammar, get_all_node_type_str, refine_atc,\
    process_source, re_organize_candidates, print_bleu_res_to_file
from translate_token_only import get_edit_dist
from util import debug


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
            'all': 'all',
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
    return _data_path, _model_base


def generate_patch(opt):
    grammar_atc = extract_atc_from_grammar(opt.grammar)
    all_node_type_seq_str = get_all_node_type_str(opt.tmp_file)
    total_number_of_test_examples = len(all_node_type_seq_str)
    all_atcs = [grammar_atc for _ in range(total_number_of_test_examples)]
    if opt.atc is not None:
        all_atcs = refine_atc(all_atcs, opt.atc)
    translator = build_translator(opt, report_score=True, multi_feature_translator=True)
    scores, all_cands = translator.translate(src_path=opt.src,
                                             tgt_path=opt.tgt,
                                             src_dir=opt.src_dir,
                                             batch_size=opt.batch_size,
                                             attn_debug=opt.attn_debug,
                                             node_type_seq=all_node_type_seq_str,
                                             atc=all_atcs)
    beam_size = len(scores[0])
    exp_name = opt.name
    all_sources = []
    all_targets = []
    tgt_file = open(opt.tgt)
    src_file = open(opt.src)
    for a, b in zip(src_file, tgt_file):
        all_sources.append(process_source(a.strip()))
        all_targets.append(process_source(b.strip()))
    tgt_file.close()
    src_file.close()
    correct = 0
    no_change = 0
    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists('result_eds'):
        os.mkdir('result_eds')

    decode_res_file = open('results/' + exp_name + '_' + str(beam_size) + '_decode_res.txt', 'w')
    bleu_file = open('result_eds/' + exp_name + '_' + str(beam_size) + '_bleus.csv', 'w')

    all_eds = []
    total_example = 0
    for idx, (src, tgt, cands) in enumerate(zip(all_sources, all_targets, all_cands)):
        atc = all_atcs[idx]
        total_example += 1
        decode_res_file.write(str(idx) + '\n')
        decode_res_file.write(src + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------------\n')
        decode_res_file.write(tgt + '\n')
        if src == tgt:
            no_change += 1
        decode_res_file.write('=====================================================================================\n')
        decode_res_file.write('Canditdate Size : ' + str(len(cands)) + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------------\n')
        eds = []
        found = False
        cands_reformatted = re_organize_candidates(cands, src, opt.n_best)
        for cand in cands_reformatted:
            ed = get_edit_dist(tgt, cand)
            if cand == tgt:
                found = True
            eds.append(ed)
            decode_res_file.write(cand + '\n')
            decode_res_file.write(str(ed) + '\n')
        if found:
            correct += 1
        all_eds.append(eds)
        decode_res_file.write(str(found) + '\n\n')
        decode_res_file.flush()

    all_eds = np.asarray(all_eds)
    print_bleu_res_to_file(bleu_file, all_eds)
    decode_res_file.close()
    bleu_file.close()
    print(correct, no_change, total_example)


if __name__ == '__main__':
    dataset = sys.argv[1]
    tree_count = '2'
    if len(sys.argv) > 2:
        tree_count = sys.argv[2]
    data_path, model_base = get_paths(dataset)
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
    debug(token_options)
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
