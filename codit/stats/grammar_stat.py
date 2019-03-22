import sys, os
import pickle
import numpy as np

from codit.grammar import JavaGrammar , Rule
from util import debug

if __name__ == '__main__':
    grammar_path = '/home/saikatc/Research/OpenNMT-py/' \
                   'c_data/raw/all/concrete/grammar.bin'
    f = open(grammar_path, 'rb')
    grammar = pickle.load(f)
    f.close()
    assert isinstance(grammar, JavaGrammar)
    rules = grammar.rules
    nts = set()
    for rule in rules:
        assert isinstance(rule, Rule)
        nts.add(rule.parent)

    debug(len(nts))
    num_rules = []
    child_length_all = []
    for nt in nts:
        nt_rules = grammar[nt]
        childs_lengths = []
        for rule in nt_rules:
            len_childs = len(rule.children)
            childs_lengths.append(len_childs)
            child_length_all.append(len_childs)
        avg_child = np.mean(childs_lengths)
        debug(avg_child)
        # child_length_all.append(avg_child)
        debug(nt, '->',  len(nt_rules))
        num_rules.append(len(nt_rules))

    debug(np.mean(child_length_all))
    debug(num_rules)
    mean = np.mean(num_rules)
    median = np.median(num_rules)
    debug(mean, median)
    import matplotlib.pyplot as plt

    n, bins, patches = plt.hist(x=num_rules, bins='auto', color='#0504aa',
                                  alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
