#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import numpy as np
from nltk.translate import bleu_score


from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

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

        blue = bleu_score.sentence_bleu([ref], hyp ,
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

def main(opt):
    translator = build_translator(opt, report_score=True)
    scores, all_cands = translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)
    beam_size = len(scores[0])
    exp_name = opt.name
    all_sources = []
    all_targets = []
    tgt_file = open(opt.tgt)
    src_file = open(opt.src)
    for a, b in zip(src_file, tgt_file):
        all_sources.append(a.strip())
        all_targets.append(b.strip())
    tgt_file.close()
    src_file.close()
    correct = 0
    no_change = 0
    decode_res_file = open('results/' + exp_name + '_' + str(beam_size) + '_decode_res.txt', 'w')
    bleu_file = open('result_bleus/' + exp_name + '_'+ str(beam_size) + '_bleus.csv', 'w')

    all_bleus = []
    total_example = 0
    for idx, (src, tgt, cands) in enumerate(zip(all_sources, all_targets, all_cands)):
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
        bleus = []
        found = False
        for cand in cands:
            bleu = get_bleu_score([tgt], [cand])
            if cand == tgt:
                found = True
            bleus.append(bleu)
            decode_res_file.write(cand + '\n')
            decode_res_file.write(str(bleu) + '\n')
        if found:
            correct += 1
        all_bleus.append(bleus)
        decode_res_file.write(str(found) + '\n\n')

    all_bleus = np.asarray(all_bleus)
    print_bleu_res_to_file(bleu_file, all_bleus)
    decode_res_file.close()
    bleu_file.close()
    print(correct, no_change, total_example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    parser.add_argument('--name', help='Name of the Experiment')

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
