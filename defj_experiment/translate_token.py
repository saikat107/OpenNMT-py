#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import copy
import os

import argparse
import pickle

import numpy as np
from nltk.translate import bleu_score

from codit.grammar import JavaGrammar
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator


import onmt.opts
from translate_structure import get_edit_dist
from util import debug


def get_bleu_score(original_codes, generated_top_result):
    blue_scores = []
    for reference, hypothesis in zip(original_codes, generated_top_result):
        ref = []
        for x in reference.split(' '):
            if x.strip() != '':
                ref.append(x.strip())
        hyp = []
        for x in hypothesis.split(' '):
            if x.strip() != '':
                hyp.append(x.strip())

        blue = bleu_score.sentence_bleu([ref], hyp,
                                        smoothing_function=bleu_score.SmoothingFunction().method3)
        blue_scores.append(blue)
    return blue_scores


def print_bleu_res_to_file(b_file, bls):
    if isinstance(bls, np.ndarray):
        r = bls.shape[0]
        for i in range(r):
            s = ','.join([str(x) for x in bls[i]])
            b_file.write(s + '\n')
        first_cand_bleus = [x[0] if len(x) > 0 else 0.0 for x in bls ]
        avg_cand_bleus = [np.mean(x) if len(x) > 0 else 0.0 for x in bls]
        cand_max_bleus = [np.max(x) if len(x) > 0 else 0.0 for x in bls]
        #print np.mean(first_cand_bleus), np.mean(avg_cand_bleus), np.mean(cand_max_bleus)
        return np.mean(first_cand_bleus), np.mean(avg_cand_bleus), np.mean(cand_max_bleus)
    pass


def get_all_node_type_str(file_path):
    node_types, all_scores = [], []
    with open(file_path) as inp:
        for line in inp:
            line = line.strip()
            parts = line.split('\t')
            node_sequences = []
            scores = []
            for part in parts:
                score_parts = part.split('/')
                node_sequences.append(score_parts[0])
                if len(score_parts) > 1:
                    scores.append(float(score_parts[1]))
                else:
                    scores.append(1.0)
            node_types.append(node_sequences)
            all_scores.append(scores)
        inp.close()
        return node_types, all_scores
    pass


def process_source(src_str):
    src_str = src_str.strip()
    words = src_str.split()
    src_modified = [word.split(u"|")[0] for word in words]
    return ' '.join(src_modified)
    pass


def re_organize_candidates(cands, scores, src, n_best):
    all_scores = []
    for cand, score in zip(cands, scores):
        all_scores.append(score)
    sorted_indices = np.argsort(all_scores)
    reformatted = []
    for i in sorted_indices:#[:n_best]:
        reformatted.append(cands[sorted_indices[i]])
    return reformatted
    # dists =
    pass


def extract_atc_from_grammar(grammar_file):
    f = open(grammar_file, 'rb')
    grammar = pickle.load(f)
    f.close()
    assert isinstance(grammar, JavaGrammar)
    value_node_rules = grammar.value_node_rules
    return value_node_rules
    pass


def refine_atc(all_atcs, atc_file):
    f = open(atc_file, 'rb')
    code_atcs = pickle.load(f)
    f.close()
    new_atcs = []
    for i in range(len(all_atcs)):
        atc = copy.copy(all_atcs[i])
        # debug(len(atc['40']))
        atc['40'] = [token for token in code_atcs[i]['40']]
        atc['800'] = [token for token in code_atcs[i]['800']]
        atc['801'] = [token for token in code_atcs[i]['801']]
        atc['802'] = [token for token in code_atcs[i]['802']]
        # debug(len(atc['40']))
        new_atcs.append(atc)
        # debug('')
    return new_atcs
    pass


def main(opt):
    # # This is just a dummy to test the implementation
    # # TODO this needs to be fixed
    # write_dummy_generated_node_types(opt.tgt, 'tmp/generated_node_types.nt')
    # ####################################################################################
    #TODO 1. Extract grammar and build initial atc
    #TODO 2. Extract atc_file and enhance atc

    grammar_atc = extract_atc_from_grammar(opt.grammar)
    all_node_type_seq_str, node_seq_scores = get_all_node_type_str(opt.tmp_file)
    total_number_of_test_examples = len(all_node_type_seq_str)
    all_atcs = [grammar_atc for _ in range(total_number_of_test_examples)]
    if opt.atc is not None:
        all_atcs = refine_atc(all_atcs, opt.atc)
    # exit()
    # for i, atc in enumerate(all_atcs):
    #     for key in atc.keys():
    #         debug(i, key, len(atc[key]))
    #     exit()
    #     debug('')
    all_file_set = set()
    correct_cand_file_set = set()
    translator = build_translator(opt, report_score=True, multi_feature_translator=True)
    all_scores, all_cands = translator.translate(src_path=opt.src,
                                             tgt_path=opt.tgt,
                                             src_dir=opt.src_dir,
                                             batch_size=opt.batch_size,
                                             attn_debug=opt.attn_debug,
                                             node_type_seq=[all_node_type_seq_str, node_seq_scores],
                                             atc=all_atcs)
    beam_size = len(all_scores[0])
    exp_name = opt.name
    all_sources = []
    all_targets = []
    all_files, all_parent_trees, all_child_trees = [], [], []
    tgt_file = open(opt.tgt)
    src_file = open(opt.src)
    files_file = open(opt.files_file)
    parent_tree_file = open(opt.parent_tree)
    child_tree_file = open(opt.child_tree)
    for src, tgt, file_path, parent_tree, child_tree in \
            zip(src_file, tgt_file, files_file, parent_tree_file, child_tree_file):
        all_sources.append(process_source(src.strip()))
        all_targets.append(process_source(tgt.strip()))
        all_files.append(file_path.strip())
        all_parent_trees.append(parent_tree.strip())
        all_child_trees.append(child_tree.strip())
    tgt_file.close()
    src_file.close()
    files_file.close()
    parent_tree_file.close()
    child_tree_file.close()
    correct = 0
    no_change = 0
    if not os.path.exists('defj_experiment/results'):
        os.mkdir('defj_experiment/results')

    if not os.path.exists('defj_experiment/result_eds'):
        os.mkdir('defj_experiment/result_eds')

    decode_res_file = open('defj_experiment/results/' + exp_name + '_' + str(beam_size) + '_decode_res.txt', 'w')
    bleu_file = open('defj_experiment/result_eds/' + exp_name + '_' + str(beam_size) + '_bleus.csv', 'w')

    all_eds = []
    total_example = 0
    for idx, (src, tgt, file_path, parent_tree, child_tree, cands, scores) in \
            enumerate(zip(all_sources, all_targets, all_files,
                          all_parent_trees, all_child_trees, all_cands, all_scores)):
        decode_res_file.write('========================================================================\n')
        total_example += 1
        decode_res_file.write('Example Number: ' + str(idx + 1) + '\n')
        decode_res_file.write('Parent Code is: \n' + src + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------\n')
        decode_res_file.write('Child code is: \n' + tgt + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------\n')
        decode_res_file.write('Parent Tree is : \n' + str(parent_tree) + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------\n')
        decode_res_file.write('Child Tree is : \n' + str(child_tree) + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------\n')
        decode_res_file.write(str(file_path) + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------\n')
        decode_res_file.write('Edit Distance : ' + str(get_edit_dist(src, tgt)) + '\n')
        decode_res_file.write('-------------------------------------------------------------------------------\n')

        if src == tgt:
            no_change += 1

        eds = []
        found = False
        cands_reformatted = re_organize_candidates(cands, scores, src, opt.n_best)
        decode_res_file.write('Cadidate List Length : ' + str(len(cands_reformatted)) + '\n')
        debug(idx, 'Cadidate List Length : ' + str(len(cands_reformatted)))
        # print(len(cands_reformatted))
        for cand in cands_reformatted:
            ed = get_edit_dist(tgt, cand)
            if cand == tgt:
                found = True
            eds.append(ed)

        fn = str(file_path)
        parts_of_fn = fn.split('/')
        if 'parent' in parts_of_fn:
            pidx = parts_of_fn.index('parent')
            project_name = parts_of_fn[pidx - 2]
            bugid = parts_of_fn[pidx - 1]
            project_bug_id = project_name + " " + bugid
        else:
            project_bug_id = fn

        all_file_set.add(project_bug_id)
        if found:
            correct_cand_file_set.add(project_bug_id)
            print(project_bug_id)
            print(src)
            print(tgt)
            decode_res_file.write("Correct\n")
            correct += 1
        else:
            decode_res_file.write("Wrong\n")
        decode_res_file.write('========================================================================\n')

        all_eds.append(eds)
        decode_res_file.write(str(found) + '\n\n')
        decode_res_file.flush()
        for cid, cand in enumerate(cands_reformatted):
            code_s = cand
            tree_s = code_s
            decode_res_file.write(str(cid) + '\nTree\t' + str(tree_s) + '\nCode\t' +
                         code_s + '\nDistance : ' + str(eds[cid]) + '\n\n')
        decode_res_file.write('========================================================================\n\n\n')


    all_eds = np.asarray(all_eds)
    print_bleu_res_to_file(bleu_file, all_eds)
    decode_res_file.close()
    bleu_file.close()
    print(correct, no_change, total_example, len(all_file_set), len(correct_cand_file_set))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='translate_token.py',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     onmt.opts.add_md_help_argument(parser)
#     onmt.opts.translate_opts(parser)
#     parser.add_argument('--name', help='Name of the Experiment')
#     parser.add_argument('--tmp_file', default='')
#     parser.add_argument('--grammar', required=True)
#     parser.add_argument('--atc', default=None)
#     parser.add_argument('--tree_count', type=int, default=2)
#
#     opt = parser.parse_args()
#     opt.batch_size = 1
#     logger = init_logger(opt.log_file)
#     main(opt)
