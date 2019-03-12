import argparse

import onmt


def get_structure_transformation_parser():
    parser_structure = argparse.ArgumentParser(
        description='full_translation.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser_structure)
    onmt.opts.translate_opts(parser_structure)
    parser_structure.add_argument('--grammar', help='Path of the grammar file', required=True)
    parser_structure.add_argument('-tmp_file', default='tmp/generated_node_types.nt')
    return parser_structure


def get_token_transformation_parser():
    parser = argparse.ArgumentParser(
        description='translate_token.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    parser.add_argument('--name', help='Name of the Experiment')
    return parser


def get_options(options):
    structure_options = get_structure_transformation_parser().parse_args(
        (' -src ' + options.src_struct + ' -batch_size 64'
         + ' -model ' + options.model_structure + ' -beam_size 100 -n_best 100 -gpu 0 '
         + ' --grammar ' + options.grammar  + ' -tmp_file tmp/' + options.name + '.gen.rule -verbose').split())
    token_options = get_token_transformation_parser().parse_args(
        ('-gpu 0 -model ' + options.model_token + ' -src ' + options.src_token + ' -tgt ' + options.tgt_token
         + ' --name ' + options.name + ' -batch_size 1 ' + ' -beam_size ' + str(options.beam_size)
         + ' -n_best ' + str(options.n_best) + ' -verbose' )
            .split())
    return structure_options, token_options
