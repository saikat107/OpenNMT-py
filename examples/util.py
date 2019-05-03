import nltk
from apted import Config, APTED

from codit.grammar import ASTNode


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
            root.value = child_value
            root.children = []
        else:
            for child in root.children:
                fix_ast(child)


def create_tree_from_string(line):
    tokens = line.strip().split(' ')
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


def read_patch_category(patch_classification_file):
    category_dict = {}
    patch_to_category_dict = {}
    with open(patch_classification_file) as f:
        pidx = -1
        category = ''
        p_code = ''
        c_code = ''
        for ln, line in enumerate(f):
            if ln % 7 == 0:
                pidx = int(line.split(':')[1].strip())
            elif ln %7 == 1:
                p_code = line.strip()
            elif ln % 7 ==2:
                c_code = line.strip()
            elif ln % 7 == 4:
                category = line.split(':')[1].strip()
                category_dict[pidx] = category
                patch_to_category_dict[p_code + c_code] = category
        f.close()
    return category_dict, patch_to_category_dict
    pass


def count_number_of_nodes(tree):
    assert isinstance(tree, ASTNode)
    count = 0
    for child in tree.children:
        count += count_number_of_nodes(child)
    return 1 + count
    pass