import sys, os

from codit.grammar import ASTNode, get_grammar

import numpy as np
import argparse
import pickle


import os
def debug(*msg):
    import inspect
    file_path = inspect.stack()[1][1]
    line_num = inspect.stack()[1][2]
    file_name = file_path
    if os.getcwd() in file_path:
        file_name = file_path[len(os.getcwd())+1:]
    stack = str(file_name) + ' # ' + str(line_num)
    print(stack, end=' ')
    res = '\t'
    for ms in msg:
        res += (str(ms) + ' ')
    print(res)


def serialize_to_file(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    f = open(path, 'wb')
    pickle.dump(obj, f, protocol=protocol)
    f.close()


def deserialize_from_file(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def read_code_from_file(file):
    codes = []
    with open(file) as fp:
        for line in fp:
            codes.append(line.strip())
    return codes


def read_binary_tree_string_from_file(file):
    trees = []
    with open(file) as fp:
        for line in fp:
            trees.append(line.strip())
    return trees


value_nodes = []


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


def read_and_create_ast_from_file(file):
    trees = []
    with open(file) as fp:
        for line in fp:
            line = line.strip()
            # print line
            trees.append(create_tree_from_string(line))
    return trees


def get_left_most_node(node):
    assert isinstance(node, ASTNode)
    if len(node.children) != 0:
        return get_left_most_node(node.children[0])
    else:
        return node


def get_right_most_node(node):
    assert isinstance(node, ASTNode)
    if len(node.children) != 0:
        return get_right_most_node(node.children[-1])
    else:
        return node


def find_closest_right_cousin(node, tree):
    assert isinstance(node, ASTNode) and isinstance(tree, ASTNode)
    if node.parent is None:
        return None
    parent = node.parent
    number_of_siblings = len(parent.children)
    if node not in parent.children:
        return None
    ci = parent.children.index(node)
    if ci < number_of_siblings - 1:
        return get_left_most_node(parent.children[ci+1])
    else:
        return find_closest_right_cousin(parent, tree)
    pass


def find_closest_left_cousin(node, tree):
    assert isinstance(node, ASTNode) and isinstance(tree, ASTNode)
    if node.parent is None:
        return None
    parent = node.parent
    if node not in parent.children:
        return None
    ci = parent.children.index(node)
    if ci > 0:
        return get_right_most_node(parent.children[ci - 1])
    else:
        return find_closest_left_cousin(parent, tree)
    pass


def get_identifier_type(node, tree, code):
    assert isinstance(node, ASTNode)
    left_cousin = find_closest_left_cousin(node, tree)
    right_cousin = find_closest_right_cousin(node, tree)
    if left_cousin is not None and left_cousin.value == 'new':
        return 'type'
    if right_cousin is not None:
        if right_cousin.value == '(' or right_cousin.value == '()' or right_cousin.value == '( )':
            return 'method'
        if right_cousin.value == '[' or right_cousin.value == '[]':
            return 'type'
        if right_cousin.type == '42':
            return 'type'
    pass


def pre_process_java_change_data(parent_codes, parent_trees, child_codes,
                                child_trees, parent_tree_os, file_names=None,
                                child_original_codes=None, allowed_tokens_file=None, type='original'):
    data = []
    if allowed_tokens_file is None:
        allowed_tokens_file = [None] * len(parent_codes)
    if file_names is None:
        file_names = [None] * len(parent_codes)
    if child_original_codes is None:
        for idx, (parent_code, parent_tree, child_code, child_tree,
                  parent_tree_o, allowed_tokens, file_name) in \
                enumerate(zip(parent_codes, parent_trees, child_codes, child_trees,
                              parent_tree_os, allowed_tokens_file, file_names)):
            if parent_tree is None or len(parent_tree) < 5:
                continue
            p_code_splitted = parent_code.split()
            c_code_splitted = child_code.split()
            if type != 'original':
                assert isinstance(parent_tree_o, ASTNode) and isinstance(child_tree, ASTNode)
                variables = set()
                method_names = set()
                type_names = set()
                packages = set()
                p_nodes = parent_tree_o.get_leaves()
                for node in p_nodes:
                    if node.type.strip() == '42':
                        ident_type = get_identifier_type(node, parent_tree_o, parent_code)
                        if ident_type == 'method':
                            method_names.add(node.value.strip())
                        elif ident_type == 'type':
                            type_names.add(node.value.strip())
                        else:
                            variables.add(node.value.strip())
                    elif node.type.strip() == '40':
                        packages.add(node.value.strip())
                c_nodes = child_tree.get_leaves()
                for node in c_nodes:
                    if node.type.strip() == '42' :
                        ident_type = get_identifier_type(node, child_tree, child_code)
                        if ident_type == 'method':
                            method_names.add(node.value.strip())
                        elif ident_type == 'type':
                            type_names.add(node.value.strip())
                        else:
                            variables.add(node.value.strip())
                    elif node.type.strip() == '40':
                        packages.add(node.value.strip())
                variable_map = {}
                for idx, v in enumerate(variables):
                    variable_map[v] = "VAR_" + str(idx + 1)
                method_name_map = {}
                for idx, v in enumerate(method_names):
                    method_name_map[v] = "METHOD_" + str(idx + 1)
                type_map = {}
                for idx, v in enumerate(type_names):
                    type_map[v] = "TYPE_" + str(idx + 1)
                package_map = {}
                for idx, v in enumerate(packages):
                    package_map[v] = "PACKAGE_" + str(idx + 1)

                for idx, token in enumerate(p_code_splitted):
                    token = token.strip()
                    if token in type_map.keys():
                        p_code_splitted[idx] = type_map[token]
                    elif token in variable_map.keys():
                        p_code_splitted[idx] = variable_map[token]
                    elif token in method_name_map.keys():
                        p_code_splitted[idx] = method_name_map[token]
                    elif token in package_map.keys():
                        p_code_splitted[idx] = package_map[token]
                for idx, token in enumerate(c_code_splitted):
                    token = token.strip()
                    if token in type_map.keys():
                        c_code_splitted[idx] = type_map[token]
                    elif token in variable_map.keys():
                        c_code_splitted[idx] = variable_map[token]
                    elif token in method_name_map.keys():
                        c_code_splitted[idx] = method_name_map[token]
                    elif token in package_map.keys():
                        c_code_splitted[idx] = package_map[token]
                for idx, token in enumerate(allowed_tokens):
                    token = token.strip()
                    if token in type_map.keys():
                        allowed_tokens[idx] = type_map[token]
                    elif token in variable_map.keys():
                        allowed_tokens[idx] = variable_map[token]
                    elif token in method_name_map.keys():
                        allowed_tokens[idx] = method_name_map[token]
                    elif token in package_map.keys():
                        allowed_tokens[idx] = package_map[token]
                child_code = ' '.join(c_code_splitted)
                # allowed_tokens = ' '.join(allowed_tokens_splitted)
                for node in p_nodes:
                    if node.type.strip() == '40' or node.type.strip() == '42':
                        v = node.value.strip()
                        if v in variable_map.keys():
                            node.value = variable_map[v]
                        elif v in type_map.keys():
                            node.value = type_map[v]
                        elif v in method_name_map.keys():
                            node.value = method_name_map[v]
                        elif v in package_map.keys():
                            node.value = package_map[v]
                for node in c_nodes:
                    if node.type.strip() == '40' or node.type.strip() == '42':
                        v = node.value.strip()
                        if v in variable_map.keys():
                            node.value = variable_map[v]
                        elif v in type_map.keys():
                            node.value = type_map[v]
                        elif v in method_name_map.keys():
                            node.value = method_name_map[v]
                        elif v in package_map.keys():
                            node.value = package_map[v]

            example = {'id': idx, 'query_tokens': p_code_splitted, 'code': child_code,
                       'parent_tree': parent_tree,
                       'child_tree': child_tree,
                       'parent_original_tree': parent_tree_o,
                       'allowed_tokens': allowed_tokens,
                       'file_name': file_name}

            data.append(example)
    else:
        for idx, (parent_code, parent_tree, child_code, child_tree, parent_tree_o,
                  child_o_code, allowed_tokens, file_name) in \
                enumerate(zip(parent_codes, parent_trees, child_codes, child_trees, parent_tree_os,
                              child_original_codes, allowed_tokens_file, file_names)):
            if parent_tree is None or len(parent_tree) < 5:
                continue
            example = {'id': idx, 'query_tokens': parent_code.split(), 'code': child_code,
                       'parent_tree': parent_tree, 'child_tree': child_tree,
                       'parent_original_tree': parent_tree_o,
                       'child_original_code' : child_o_code,
                       'allowed_tokens': allowed_tokens,
                       'file_name': file_name}

            data.append(example)

    return data


# helper function begins
def get_terminal_tokens(_terminal_str):
    # tmp_terminal_tokens = _terminal_str.split(' ')
    # _terminal_tokens = []
    # for token in tmp_terminal_tokens:
    #     if token:
    #         _terminal_tokens.append(token)
    #     _terminal_tokens.append(' ')
    #
    # return _terminal_tokens[:-1]
    return [_terminal_str.strip()]


def read_raw_data(p_code, p_tree, c_code, c_tree, parent_original_tree,
                  allowed_tokens_file, file_names_file, exc_str_ch=False, samples=10000000):
    pc = open(p_code)
    pt = open(p_tree)
    cc = open(c_code)
    ct = open(c_tree)
    pot = open(parent_original_tree)
    ats = open(allowed_tokens_file)
    fnf = None
    if os.path.exists(file_names_file):
        fnf = open(file_names_file)

    included_ids = []
    pcodes = []
    ptrees = []
    ccodes = []
    ctrees = []
    potrees = []
    allowed_tokens = []
    file_names = []
    if fnf is None:
        counter = 0
        for idx, (pcs, pts, ccs, cts, pots, at) in enumerate(zip(pc, pt, cc, ct, pot, ats)):
            if exc_str_ch and cts.strip() == pots.strip():
                continue
            counter += 1
            if '`' in ccs:
                continue
            if len(pts.strip()) < 5:
                continue
            pcodes.append(pcs.strip())
            ptrees.append(pts.strip())
            ccodes.append(ccs.strip())
            ctrees.append(create_tree_from_string(cts))
            potrees.append(create_tree_from_string(pots))
            allowed_tokens.append([s.strip() for s in at.split()])
            included_ids.append(idx)
            if counter == samples:
                break
        file_names = [None] * len(pcodes)
    else:
        counter = 0
        for idx, (pcs, pts, ccs, cts, pots, at, fn) in enumerate(zip(pc, pt, cc, ct, pot, ats, fnf)):
            #if exc_str_ch and cts.strip() == pots.strip():
            #    continue
            counter += 1
            if '`' in ccs:
                continue
            if len(pts.strip()) < 5:
                continue
            pcodes.append(pcs.strip())
            ptrees.append(pts.strip())
            ccodes.append(ccs.strip())
            ctrees.append(create_tree_from_string(cts))
            potrees.append(create_tree_from_string(pots))
            allowed_tokens.append([s.strip() for s in at.split()])
            file_names.append(fn)
            included_ids.append(idx)
            if counter == samples:
                break
    return pcodes[:samples], ptrees[:samples], ccodes[:samples], \
           ctrees[:samples], potrees[:samples], allowed_tokens[:samples], file_names[:samples], included_ids[:samples]


def read_raw_data_for_type_data(p_code, p_tree, c_code, c_tree, parent_original_tree,
                  child_original_code, allowed_tokens_file, exc_str_ch=False, samples=10000000):
    pc = open(p_code)
    pt = open(p_tree)
    cc = open(c_code)
    ct = open(c_tree)
    pot = open(parent_original_tree)
    choc = open(child_original_code)
    ats = open(allowed_tokens_file)
    included_ids = []
    pcodes = []
    ptrees = []
    ccodes = []
    ctrees = []
    potrees = []
    cocodes = []
    allowed_tokens = []
    counter = 0
    for idx, (pcs, pts, ccs, cts, pots, chc, at) in enumerate(zip(pc, pt, cc, ct, pot, choc, ats)):
        if exc_str_ch and pcs.strip() == ccs.strip():
            continue
        counter += 1
        if '`' in ccs:
            continue
        if len(pts.strip()) < 5:
            continue
        pcodes.append(pcs.strip())
        ptrees.append(pts.strip())
        ccodes.append(ccs.strip())
        ctrees.append(create_tree_from_string(cts))
        potrees.append(create_tree_from_string(pots))
        cocodes.append(chc.strip())
        allowed_tokens.append([s.strip() for s in at.split()])
        included_ids.append(idx)
        if counter == samples:
            break
    return pcodes[:samples], ptrees[:samples], ccodes[:samples], ctrees[:samples], \
           potrees[:samples], cocodes[:samples], allowed_tokens[:samples], included_ids


def get_final_token_mask(allowed_tokens, t_vocab):
    indices = []
    for word in allowed_tokens:
        if word in t_vocab.token_id_map.keys():
            indices.append(t_vocab.token_id_map[word])
        else:
            if word.strip() in t_vocab.token_id_map.keys():
                indices.append(t_vocab.token_id_map[word.strip()])
    mask = np.array([1e-30] * t_vocab.size, dtype='float32')
    for index in indices:
        mask[index] = 1.00
    # print np.sum(mask)
    return np.array(indices), mask
    pass


def read_train_and_test_data(train_folders, valid_folder, test_folder):
    parent_codes = []
    parent_trees = []
    child_codes = []
    child_trees = []
    parent_tree_o = []
    allowed_tokens_all = []
    train_ids = []
    file_names = []
    valid_ids = []
    for train_folder in train_folders:
        parent_version_code_file = train_folder + '/parent.code'
        parent_version_tree_file = train_folder + '/parent.tree'
        child_version_code_file = train_folder + '/child.code'
        child_version_tree_file = train_folder + '/child.tree'
        parent_original_tree_file = train_folder + '/parent.org.tree'
        allowed_tokens_file_name = train_folder + '/allowed.tokens'
        file_names_file = train_folder + '/files.txt'
        parent_codes_train, parent_trees_train, child_codes_train, child_trees_train, \
        parent_tree_o_train, allowed_tokens_train, file_names_all, all_ids_train = read_raw_data(
            parent_version_code_file,
            parent_version_tree_file,
            child_version_code_file,
            child_version_tree_file,
            parent_original_tree_file,
            allowed_tokens_file_name,
            file_names_file,
            True
        )
        parent_codes.extend(parent_codes_train)
        parent_trees.extend(parent_trees_train)
        child_codes.extend(child_codes_train)
        child_trees.extend(child_trees_train)
        parent_tree_o.extend(parent_tree_o_train)
        allowed_tokens_all.extend(allowed_tokens_train)
        file_names.extend(file_names_all)
        train_ids.extend(all_ids_train)

    parent_version_code_file = valid_folder + '/parent.code'
    parent_version_tree_file = valid_folder + '/parent.tree'
    child_version_code_file = valid_folder + '/child.code'
    child_version_tree_file = valid_folder + '/child.tree'
    parent_original_tree_file = valid_folder + '/parent.org.tree'
    allowed_tokens_file_name = valid_folder + '/allowed.tokens'
    file_names_file = valid_folder + '/files.txt'
    parent_codes_test, parent_trees_test, child_codes_test, child_trees_test, \
    parent_tree_o_test, allowed_tokens_test, file_names_all, all_ids_test = read_raw_data(
        parent_version_code_file,
        parent_version_tree_file,
        child_version_code_file,
        child_version_tree_file,
        parent_original_tree_file,
        allowed_tokens_file_name,
        file_names_file,
        True
    )
    parent_codes.extend(parent_codes_test)
    parent_trees.extend(parent_trees_test)
    child_codes.extend(child_codes_test)
    child_trees.extend(child_trees_test)
    parent_tree_o.extend(parent_tree_o_test)
    allowed_tokens_all.extend(allowed_tokens_test)
    file_names.extend(file_names_all)
    valid_ids.extend(all_ids_test)

    parent_version_code_file = test_folder + '/parent.code'
    parent_version_tree_file = test_folder + '/parent.tree'
    child_version_code_file = test_folder + '/child.code'
    child_version_tree_file = test_folder + '/child.tree'
    parent_original_tree_file = test_folder + '/parent.org.tree'
    allowed_tokens_file_name = test_folder + '/allowed.tokens'
    file_names_file = test_folder + '/files.txt'
    parent_codes_test, parent_trees_test, child_codes_test, child_trees_test, \
    parent_tree_o_test, allowed_tokens_test, file_names_all, all_ids_test = read_raw_data(
        parent_version_code_file,
        parent_version_tree_file,
        child_version_code_file,
        child_version_tree_file,
        parent_original_tree_file,
        allowed_tokens_file_name,
        file_names_file,
        True
    )
    parent_codes.extend(parent_codes_test)
    parent_trees.extend(parent_trees_test)
    child_codes.extend(child_codes_test)
    child_trees.extend(child_trees_test)
    parent_tree_o.extend(parent_tree_o_test)
    allowed_tokens_all.extend(allowed_tokens_test)
    file_names.extend(file_names_all)
    return parent_codes, parent_trees, child_codes, child_trees, parent_tree_o, \
           allowed_tokens_all, file_names, len(train_ids), len(valid_ids)


def create_all_files(folder_name, data_type):
    if not os.path.exists(folder_name + '/' + data_type):
        os.mkdir(folder_name + '/' + data_type)
    prev_rule_file = open(os.path.join(folder_name + '/' + data_type, 'prev.rule'), 'w')
    next_rule_file = open(os.path.join(folder_name+ '/' + data_type, 'next.rule'), 'w')
    prev_rule_parent_file = open(os.path.join(folder_name + '/' + data_type, 'prev.parent.rule'), 'w')
    next_rule_parent_file = open(os.path.join(folder_name + '/' + data_type, 'next.parent.rule'), 'w')
    prev_rule_parent_t_file = open(os.path.join(folder_name + '/' + data_type, 'prev.parent.time'), 'w')
    next_rule_parent_t_file = open(os.path.join(folder_name + '/' + data_type, 'next.parent.time'), 'w')
    prev_token_node_id_file = open(os.path.join(folder_name + '/' + data_type, 'prev.token.id'), 'w')
    next_token_node_id_file = open(os.path.join(folder_name + '/' + data_type, 'next.token.id'), 'w')
    prev_token_file = open(os.path.join(folder_name + '/' + data_type, 'prev.token'), 'w')
    next_token_file = open(os.path.join(folder_name + '/' + data_type, 'next.token'), 'w')
    prev_token_plus_id_file = open(os.path.join(folder_name + '/' + data_type, 'prev.augmented.token'), 'w')
    next_token_plus_id_file = open(os.path.join(folder_name + '/' + data_type, 'next.augmented.token'), 'w')
    return prev_rule_file, next_rule_file, prev_rule_parent_file, next_rule_parent_file, \
    prev_rule_parent_t_file, next_rule_parent_t_file, prev_token_node_id_file, \
    next_token_node_id_file, prev_token_file, next_token_file, prev_token_plus_id_file, next_token_plus_id_file


def write_contents(prev_rule_file, next_rule_file, prev_rule_parent_file, next_rule_parent_file,
    prev_rule_parent_t_file, next_rule_parent_t_file, prev_token_node_id_file,
    next_token_node_id_file, prev_token_file, next_token_file, prev_token_plus_id_file, next_token_plus_id_file,
                   prev_rule, next_rule, prev_rule_parent, next_rule_parent,
    prev_rule_parent_t, next_rule_parent_t, prev_token_node_id,
    next_token_node_id, prev_token, next_token):
    prev_rule_file.write(' '.join([str(x) for x in prev_rule]) + '\n')
    next_rule_file.write(' '.join([str(x) for x in next_rule]) + '\n')
    prev_rule_parent_file.write(' '.join([str(x) for x in prev_rule_parent]) + '\n')
    next_rule_parent_file.write(' '.join([str(x) for x in next_rule_parent]) + '\n')
    prev_rule_parent_t_file.write(' '.join([str(x) for x in prev_rule_parent_t]) + '\n')
    next_rule_parent_t_file.write(' '.join([str(x) for x in next_rule_parent_t]) + '\n')
    prev_token_node_id_file.write(' '.join([str(x) for x in prev_token_node_id]) + '\n')
    next_token_node_id_file.write(' '.join([str(x) for x in next_token_node_id]) + '\n')
    prev_token_file.write(' '.join([str(x) for x in prev_token]) + '\n')
    next_token_file.write(' '.join([str(x) for x in next_token]) + '\n')
    ############### Do the alighment #######################
    prev_token_node_id.append(-1)
    next_token_node_id.append(-1)
    prev_token_node_id = prev_token_node_id[1:]
    next_token_node_id = next_token_node_id[1:]
    #######################################################
    prev_augmented_token_str = ' '.join([str(x) + u"|" + str(y) for x,y in zip(prev_token, prev_token_node_id)]) + '\n'
    next_augmented_token_str = ' '.join([str(x) + u"|" + str(y) for x, y in zip(next_token, next_token_node_id)]) + '\n'
    prev_token_plus_id_file.write(prev_augmented_token_str)
    next_token_plus_id_file.write(next_augmented_token_str)


def flush_all(*files):
    for f in files:
        f.flush()


def closs_all(*files):
    for f in files:
        f.close()


def parse_java_change_dataset():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='Main Data Directory', default='/Users/saikat')
    parser.add_argument('-source', help='Relative path of the data source', default='Research')
    parser.add_argument('-train', help='Train Folder Name(s)', nargs='+', default=['train'])
    parser.add_argument('-valid', help='Train Folder Name', default='valid')
    parser.add_argument('-test', help='Train Folder Name(s)',  default='test')

    parser.add_argument('-output', help='name of the output folder', default='rule_list_output')
    parser.add_argument('-name', help='name of the data file',default='small')
    parser.add_argument('-exclude_no_structure_change', action='store_true')


    # parser.set_defaults(exclude_string_change=True)
    args = parser.parse_args()
    data_base = os.path.join(args.data, args.source)
    name = args.name
    data_directory = os.path.join(data_base, name)
    train_folders = [os.path.join(data_directory, train) for train in args.train]
    test_folder = os.path.join(data_directory, args.test)
    valid_folder = os.path.join(data_directory, args.valid)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    parent_codes, parent_trees, child_codes, child_trees, parent_tree_o, allowed_tokens_for_nodes, \
        file_names , num_train_examples, num_valid_examples \
        = read_train_and_test_data(train_folders, valid_folder, test_folder)
    total_examples = len(parent_codes)
    num_test_examples = total_examples - num_train_examples - num_valid_examples

    assert len(parent_codes) == len(parent_trees) and len(parent_codes) == len(child_codes) and \
           len(parent_codes) == len(child_trees) and len(parent_codes) == len(parent_tree_o) and \
           len(parent_codes) == len(allowed_tokens_for_nodes)

    debug('Training Examples :\t' + str(num_train_examples))
    debug('Validation Examples :\t' + str(num_valid_examples))
    debug('Test Examples :  \t' + str(num_test_examples))
    num_valid_examples += num_train_examples

    debug('Finished Reading Data')
    grammar = get_grammar(child_trees)
    debug('Total rules : ' + str((len(grammar.rules))))
    debug(grammar.terminal_nodes)
    value_nodes = grammar.value_node_rules.keys()
    for node in value_nodes:
        debug(grammar.value_node_rules[node])
    # debug('Total terminal nodes : ' + str(len(grammar.terminal_nodes)))

    data = pre_process_java_change_data(parent_codes=parent_codes, parent_trees=parent_trees,
                                       child_codes=child_codes, child_trees=child_trees,
                                       parent_tree_os=parent_tree_o, file_names=file_names,
                                       allowed_tokens_file=allowed_tokens_for_nodes)

    train_data, dev_data, test_data, all_examples, train_ids, dev_ids, test_ids = [], [], [], [], [], [], []

    for idx, entry in enumerate(data):
        prev_token_node_id = []
        prev_token = []
        next_token_node_id = []
        next_token = []
        next_rule = []
        next_rule_parent_t = []
        next_rule_parent = []
        prev_rule = []
        prev_rule_parent = []
        prev_rule_parent_t = []

        parse_tree = entry['child_tree']
        parent_original_tree = entry['parent_original_tree']

        rule_list_next_v, rule_parents_next_v, value_nodes_next_v = parse_tree.get_productions(include_value_node=True)
        rule_list_prev_v, rule_parents_prev_v, value_nodes_prev_v = parent_original_tree.get_productions(include_value_node=True)
        actions = []
        rule_pos_map = dict()
        for rule_count, rule in enumerate(rule_list_next_v):
            if not grammar.is_value_node(rule):
                parent_rule = rule_parents_next_v[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0
                rule_pos_map[rule] = len(actions)
                action = 'Action(APPLY_RULE, d)'
                actions.append(action)
                next_rule.append(grammar.rule_to_id.get(rule))
                next_rule_parent_t.append(grammar.rule_to_id.get(parent_rule))
                next_rule_parent.append(parent_t)
            else:
                node_id = str(rule.type)
                node_value = str(rule.value)
                next_token_node_id.append(node_id)
                next_token.append(node_value)

        actions = []
        rule_pos_map = dict()
        for rule_count, rule in enumerate(rule_list_prev_v):
            if not grammar.is_value_node(rule):
                parent_rule = rule_parents_prev_v[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0
                rule_pos_map[rule] = len(actions)
                action = 'Action(APPLY_RULE, d)'
                actions.append(action)
                prev_rule.append(grammar.rule_to_id.get(rule))
                prev_rule_parent.append(grammar.rule_to_id.get(parent_rule))
                prev_rule_parent_t.append(parent_t)
            else:
                node_id = str(rule.type)
                node_value = str(rule.value)
                prev_token_node_id.append(node_id)
                prev_token.append(node_value)

        if args.exclude_no_structure_change and prev_rule == next_rule:
            continue
        example = [prev_rule, next_rule, prev_rule_parent, next_rule_parent,
                    prev_rule_parent_t, next_rule_parent_t, prev_token_node_id,
                    next_token_node_id, prev_token, next_token]
        all_examples.append(example)

        if idx < num_train_examples:
            train_data.append(example)
            train_ids.append(idx)
        elif idx < num_valid_examples:
            dev_data.append(example)
            dev_ids.append(idx)
        else:
            test_data.append(example)
            test_ids.append(idx)

    _file_all = create_all_files(args.output, 'train')
    for ex in train_data:
        af = [f for f in _file_all]
        af.extend(ex)
        write_contents(*af)
        flush_all(*_file_all)
    closs_all(*_file_all)

    _file_all = create_all_files(args.output, 'valid')
    for ex in dev_data:
        af = [f for f in _file_all]
        af.extend(ex)
        write_contents(*af)
        flush_all(*_file_all)
    closs_all(*_file_all)

    _file_all = create_all_files(args.output, 'test')
    for ex in test_data:
        af = [f for f in _file_all]
        af.extend(ex)
        write_contents(*af)
        flush_all(*_file_all)
    closs_all(*_file_all)

    serialize_to_file(grammar, os.path.join(args.output, 'grammar.bin'))
    return train_data, dev_data, test_data


if __name__ == '__main__':
    np.random.seed(1000)
    tra, dev, tes = parse_java_change_dataset()
    # print tra.get_prob_func_inputs([0])
