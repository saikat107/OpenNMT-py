from apted import APTED, Config

from codit.grammar import ASTNode
import nltk
from matplotlib import pyplot as plt

dataname = 'icse'

if dataname == 'codit':
    complete_split_data_path = \
        '/home/saikatc/Research/codit_data/complete_split_data/10_20_original/test'
    data_raw_path = '/home/saikatc/Research/OpenNMT-py/c_data/raw/all/concrete/test_new'
    correct_indices_path = 'codit-correct-patch.csv'
else:
    complete_split_data_path = \
        '/home/saikatc/Research/icse_data_concrete/all/small/test'
    data_raw_path = '/home/saikatc/Research/OpenNMT-py/icse_data/raw/all/concrete_small/test_new'
    correct_indices_path = 'icse-correct-patch.csv'

INDEX = 'INDEX'
EDIT_DISTANCE = 'EDIT_DISTANCE'
TREE_DISTANCE = 'TREE_DISTANCE'
PREV_CODE = 'PREV_CODE'
NEXT_CODE = 'NEXT_CODE'
PREV_TREE_STR = 'PREV_TREE_STR'
NEXT_TREE_STR = "NEXT_TREE_STR"
PREV_TREE = 'PREV_TREE'
NEXT_TREE = 'NEXT_TREE'
BIGRAM_JACCARD = 'BIGRAM_JACCARD'
BIGRAM_EDIT = 'BIGRAM_EDIT_DISTANCE'
TYPE_OF_CHANGE = 'TYPE_OF_CHANGE'
NORMALIZED_TREE_DISTANCE = 'NORMALIZED_TREE_DIST'


def transform_to_ngram(code, n=2):
    if len(code) <= n:
        return [' '.join(code)]
    n_grams = []
    for i in range(len(code) - n + 1):
        ng = []
        for j in range(n):
            ng.append(code[i+j])
        n_grams.append(' '.join(ng))
    return n_grams


def bigram_edit_distance(code1, code2):
    bigram1 = transform_to_ngram(code1.split(), n=2)
    bigram2 = transform_to_ngram(code2.split(), n=2)
    dist = nltk.edit_distance(bigram1 , bigram2)
    if dist == 0:
        print(bigram1, '   ----->  ', bigram2)
    return dist
    # return 0


def bigram_jaccard_distance(code1, code2):
    bigram1 = transform_to_ngram(code1.split() , n=2)
    bigram2 = transform_to_ngram(code2.split() , n=2)
    return nltk.jaccard_distance(set(bigram1), set(bigram2))


class TreeEditDistanceConfig(Config):
    def __init__(self):
        pass

    def rename(self, node1, node2):
        '''
        A node is considered renamed when its value is changed, if the type is changed it is not considered renaming
        For considering that,
        return 1 if (node1.type != node2.type) or (node1.value != node2.value) else 0
        :param node1:
        :param node2:
        :return:
        '''
        return 1 if node1.value != node2.value else 0

    def children(self, node):
        return [x for x in node.children]


def calculate_edit_distance(tree1, tree2):
    apted = APTED(tree1, tree2, TreeEditDistanceConfig())
    ed = apted.compute_edit_distance()
    return ed


def fix_ast(root):
    if isinstance(root, ASTNode):
        if root.is_leaf:
            return
        elif len(root.children) == 1 and root.children[0].is_leaf:
            child_value = root.children[0].type
            # if child_value == '<EMPTY>':
            #     root.children[0].value = child_value
            #     return
            # if child_value == '?':
            #     root.children[0].value = child_value
            #     return
            root.value = child_value
            root.children = []
        else:
            for child in root.children:
                fix_ast(child)


def create_tree_from_string(line):
    tokens = line.strip().split(' ')
    # print line
    stack = []
    for token in tokens:
        token = token.strip()
        if token == '':
            continue
        if token == '`':
            stack.append(token)
        elif token == '``':
            children = []
            top_of_stack = stack.pop()
            while top_of_stack != '`' and len(stack) != 0:
                children.append(top_of_stack)
                top_of_stack = stack.pop()
            top_of_stack = stack.pop()
            if isinstance(top_of_stack, ASTNode):
                for child in children:
                    top_of_stack.add_child(child)
            stack.append(top_of_stack)
        else:
            node = ASTNode(token)
            stack.append(node)
    root = stack.pop()
    fix_ast(root)
    return root

correct_data = {}
with open(correct_indices_path) as cip:
    for line in cip:
        parts = line.split(',')
        idx = int(parts[0].strip())
        dist = int(parts[1].strip())
        data = {
            INDEX: idx,
            EDIT_DISTANCE: dist
        }
        correct_data[idx] = data

prev_code_file = complete_split_data_path + '/parent.code'
next_code_file = complete_split_data_path + '/child.code'
previous_tree_file = complete_split_data_path + '/parent.org.tree'
next_tree_file = complete_split_data_path + '/child.tree'
tree_dictionary = {}
with open(previous_tree_file) as ptfile:
    with open(next_tree_file) as ntfile:
        with open(prev_code_file) as pcfile:
            with open(next_code_file) as ncfile:
                for (pc, cc, pt, nt) in zip(pcfile, ncfile, ptfile, ntfile):
                    pcs = ' '.join([token.strip() for token in pc.strip().split()])
                    ccs = ' '.join([token.strip() for token in cc.strip().split()])
                    pts = pt.strip()
                    nts = nt.strip()
                    key = pcs + ' -> ' + ccs
                    tree_dictionary[key] = [pts, nts]


draw_prev_token_file = data_raw_path + '/prev.token'
draw_next_token_file = data_raw_path + '/next.token'
count = 0

eds = []
tds = []
njds = []
neds = []
all_keys = []
ntds = []

# correct_classification_file = open('Correctly-predicted-patch.txt', 'w')
change_type_map = {}


def count_number_of_nodes(tree):
    assert isinstance(tree, ASTNode)
    count = 0
    for child in tree.children:
        count += count_number_of_nodes(child)
    return 1 + count
    pass


with open(draw_prev_token_file) as pcfile:
    with open(draw_next_token_file) as ncfile:
        for idx, (pc, nc) in enumerate(zip(pcfile, ncfile)):
            if idx not in correct_data.keys():
                continue
            pcs = ' '.join([token.strip() for token in pc.strip().split()])
            ccs = ' '.join([token.strip() for token in nc.strip().split()])
            correct_data[idx][PREV_CODE] = pcs
            correct_data[idx][NEXT_CODE] = ccs
            key = pcs + ' -> ' + ccs
            if key not in tree_dictionary.keys():
                continue
            pts, nts = tree_dictionary[key]
            correct_data[idx][PREV_TREE_STR] = pts
            correct_data[idx][NEXT_TREE_STR] = nts
            correct_data[idx][PREV_TREE] = create_tree_from_string(pts)
            correct_data[idx][NEXT_TREE] = create_tree_from_string(nts)
            correct_data[idx][TREE_DISTANCE] = calculate_edit_distance(
                correct_data[idx][PREV_TREE], correct_data[idx][NEXT_TREE])
            correct_data[idx][BIGRAM_EDIT] = bigram_edit_distance(pcs, ccs)
            correct_data[idx][BIGRAM_JACCARD] = bigram_jaccard_distance(pcs, ccs)
            correct_data[idx][NORMALIZED_TREE_DISTANCE] = \
                float(correct_data[idx][TREE_DISTANCE]) / count_number_of_nodes(correct_data[idx][PREV_TREE])
            # if correct_data[idx][BIGRAM_EDIT] == 0:
            #     print(pcs, '   ---->   ',  ccs)
            print(correct_data[idx][EDIT_DISTANCE], correct_data[idx][TREE_DISTANCE],
                  correct_data[idx][BIGRAM_EDIT], correct_data[idx][BIGRAM_JACCARD])
            if key in all_keys:
                continue
            else:
                all_keys.append(key)
            # correct_classification_file.write('Example : ' + str(idx) + '\n')
            # correct_classification_file.write(pcs + '\n')
            # correct_classification_file.write(ccs + '\n')
            # correct_classification_file.write(str(correct_data[idx][EDIT_DISTANCE]) +
            #                                   ' , ' +
            #                                   str(correct_data[idx][TREE_DISTANCE]) +
            #                                   ' , ' +
            #                                   str(correct_data[idx][BIGRAM_EDIT]) +
            #                                   ' , ' +
            #                                   str(correct_data[idx][BIGRAM_JACCARD]) +
            #                                   '\n'
            #                                   )
            print('Previous Version:\t', pcs)
            print('Next Version:   \t', ccs)
            print(',\t'.join(change_type_map.keys()))
            # change_type = input('Enter Change Type: ')
            # change_type = change_type.lower()
            # correct_data[idx][TYPE_OF_CHANGE] = change_type
            # if change_type in change_type_map.keys():
            #     change_type_map[change_type] += 1
            # else:
            #     change_type_map[change_type] = 1
            # correct_classification_file.write('Change Type : ' + change_type + '\n')
            # correct_classification_file.write('=====================================\n\n')
            eds.append(correct_data[idx][EDIT_DISTANCE])
            tds.append(correct_data[idx][TREE_DISTANCE])
            neds.append(correct_data[idx][BIGRAM_EDIT])
            njds.append(correct_data[idx][BIGRAM_JACCARD])
            ntds.append(correct_data[idx][NORMALIZED_TREE_DISTANCE])
            count += 1
print(count)
# correct_classification_file.close()
f1 = 'edit-size-hist-' + dataname + '.pdf'
f2 = 'edit-size-hist-edit-dist-' + dataname + '.pdf'
f3 = 'edit-size-hist-normalized-tree-norm-' + dataname + '.pdf'
print(f1, f2, f3)

plt.figure()
# plt.hist(eds, label='Edit Distance', alpha=0.5, ls='dashed', edgecolor='b', lw=3, color='b')
# plt.hist(neds, label='Bigram Edit Distance', alpha=0.5, ls='dotted', edgecolor='b', lw=3, color='r')
plt.hist(tds, label='Tree Distance', alpha=0.5, ls='solid', lw=3, color='g', edgecolor='b')
plt.legend()
plt.savefig(fname=f1)
plt.show()


plt.figure()
plt.hist(eds, label='Token Distance', alpha=0.5, ls='solid',  lw=3, color='b', edgecolor='b')
plt.savefig(fname=f2)
plt.legend()
plt.show()

plt.figure()
plt.hist(ntds, label='Normalized Tree Distance', alpha=0.5, ls='solid', edgecolor='b', lw=3, color='r')
plt.savefig(fname=f3)
plt.legend()
plt.show()





