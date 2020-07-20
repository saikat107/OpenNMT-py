import os
import sys
import json
import pickle
import shutil


def extract_project_name(file):
    parts = file.strip().split('/')
    return parts[-4].strip()
    pass


def read_all(base):
    examples = {}
    for part in ['train', 'valid', 'test']:
        directory = os.path.join(base, part)
        prev_token_file = open(os.path.join(directory, 'prev.token'))
        next_token_file = open(os.path.join(directory, 'next.token'))
        prev_rule_file = open(os.path.join(directory, 'prev.rule'))
        next_rule_file = open(os.path.join(directory, 'next.rule'))
        prev_aug_token_file = open(os.path.join(directory, 'prev.augmented.token'))
        next_aug_token_file = open(os.path.join(directory, 'next.augmented.token'))
        prev_abstract_token_file = open(os.path.join(directory, 'prev.abstract.code'))
        next_abstract_token_file = open(os.path.join(directory, 'next.abstract.code'))
        files_file = open(os.path.join(directory, 'files.txt'))
        atc_file = pickle.load(open(os.path.join(directory, 'atc_scope.bin'), 'rb'))
        for ptoken, ntoken, prule, nrule, paugtoken, naugtoken, patoken, natoken, file, atc in zip(
            prev_token_file, next_token_file, prev_rule_file, next_rule_file, prev_aug_token_file, next_aug_token_file,
            prev_abstract_token_file, next_abstract_token_file, files_file, atc_file
        ):
            project = extract_project_name(file)
            if project not in examples.keys():
                examples[project] = []
            examples[project].append({
                'prev.token': ptoken.strip(),
                'next.token': ntoken.strip(),
                'prev.rule': prule.strip(),
                'next.rule': nrule.strip(),
                'prev.augmented.token': paugtoken.strip(),
                'next.augmented.token': naugtoken.strip(),
                'prev.abstract.code': patoken.strip(),
                'next.abstract.code': natoken.strip(),
                'files.txt': file.strip(),
                'atc': atc
            })
        print(part, len(atc_file))
    return examples


def split_train_valid_test(projects):
    test_p = ['grails_grails-core',# 875
             'essentials_Essentials', #758
              'thinkaurelius_titan', # 1171
              'BuildCraft_BuildCraft' #2798
    ]
    valid_p = [
        'Graylog2_graylog2-server', # 1213'
        'FasterXML_jackson-databind', #812'
        'dropwizard_dropwizard', # 438'
        'jphp-compiler_jphp',  # 319
    ]
    train_p = [p for p in projects]
    for p in test_p:
        train_p.remove(p)
    for p in valid_p:
        train_p.remove(p)
    return train_p, valid_p, test_p
    pass


def write_examples(directory, examples):
    if not os.path.exists(directory):
        os.mkdir(directory)
    keys = list(examples[0].keys())
    keys.remove('atc')
    atcs = []
    files = {}
    for key in keys:
        files[key] = open(os.path.join(directory, key), 'w')
    for example in examples:
        for key in keys:
            files[key].write(example[key].strip() + '\n')
        atcs.append(example['atc'])
    atc_file = open(os.path.join(directory, 'atc_scope.bin'), 'wb')
    pickle.dump(atcs, atc_file)
    atc_file.close()
    for key in keys:
        files[key].close()



if __name__ == '__main__':
    total = 0
    data_dir = '../data/raw/code_change_data/'
    data = read_all(data_dir)
    projects, lengths = [], []
    for project in data:
        projects.append(project)
        lengths.append(len(data[project]))
        total += len(data[project])
    train_p, valid_p, test_p = split_train_valid_test(projects)
    train_examples, valid_examples, test_examples = [], [], []
    for p in train_p:
        train_examples.extend(data[p])
    for p in valid_p:
        valid_examples.extend(data[p])
    for p in test_p:
        test_examples.extend(data[p])
    print(len(train_examples), len(valid_examples), len(test_examples))
    out_dir = '../data/raw/cc_data_project_split'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    write_examples(os.path.join(out_dir, 'train'), train_examples)
    write_examples(os.path.join(out_dir, 'valid'), valid_examples)
    write_examples(os.path.join(out_dir, 'test'), test_examples)
    in_grammar_file = os.path.join(data_dir, 'grammar.bin')
    out_grammar_file = os.path.join(out_dir, 'grammar.bin')
    d = shutil.copyfile(in_grammar_file, out_grammar_file)
    print(d)
    pass