import os

from codit.create_transformation_data import deserialize_from_file , serialize_to_file
import numpy as np


def get_canonical_project_name(actual_project_name):
    p_name_dict = {
        'Lang': 'Lang',
        'google_closure-compiler': 'Closure',
        'jfree_jfreechart': 'Chart',
        'apache_commons-math': 'Math',
        'mockito_mockito': 'Mockito',
        'JodaOrg_joda-time': 'Time',
        'Mockito': 'Mockito',
        'Chart': 'Chart',
        'Math': 'Math',
        'apache_commons-lang': 'Lang',
        'Time': 'Time',
        'Closure': 'Closure'
    }
    return p_name_dict[actual_project_name]
    pass


def extract_file_information(_file_names):
    all_information = dict()
    all_information['projects'] = set()
    all_information['examples_in_project'] = dict()
    with open(_file_names) as inp:
        for example_idx, file_line in enumerate(inp):
            file_line = file_line.strip()
            parts = file_line.split('/')
            project_name = get_canonical_project_name(parts[-4].strip())
            all_information['projects'].add(project_name)
            if project_name not in all_information['examples_in_project'].keys():
                all_information['examples_in_project'][project_name] = []
            all_information['examples_in_project'][project_name].append(example_idx)
    return all_information
    pass


def create_appropriate_files_in_project_directory(train_dir, valid_dir, test_dir, all_projects):
    all_resular_files = dict()
    all_atc_files = dict()
    for project in all_projects:
        train_files_regular, train_files_atc = create_appropriate_file(project, train_dir)
        valid_files_regular, valid_files_atc = create_appropriate_file(project, valid_dir)
        test_files_regular, test_files_atc = create_appropriate_file(project, test_dir)
        all_resular_files[project] = [train_files_regular, valid_files_regular, test_files_regular]
        all_atc_files[project] = [train_files_atc, valid_files_atc, test_files_atc]
    return all_resular_files, all_atc_files
    pass


interesting_files = ['prev.token', 'next.token', 'prev.token.id', 'next.token.id', 'prev.augmented.token',
                     'next.augmented.token', 'files.txt', 'prev.rule', 'next.rule', 'prev.tree', 'next.tree']
atc_files = ['atc_scope.bin', 'atc_method.bin']


def create_appropriate_file(project, _dir):
    project_dir_ = os.path.join(_dir, project)
    if not os.path.exists(project_dir_):
        os.mkdir(project_dir_)
    _files_all_ = []
    for file_name in interesting_files:
        _files_all_.append(open(project_dir_ + '/' + file_name, 'w'))
    _files_atc = []
    for atc_file in atc_files:
        _files_atc.append(project_dir_ + '/' + atc_file)
    result = dict()
    result['project'] = project
    result['file_names'] = _files_all_
    return _files_all_, _files_atc


def read_data_all(_dir):
    all_files = []
    all_contents = []
    all_atcs = []
    for file_name in interesting_files:
        all_files.append(open(os.path.join(_dir, file_name)))
    atcs = []
    for atc_file in atc_files:
        atcs.append(deserialize_from_file(os.path.join(_dir, atc_file)))
    for idx, contents in enumerate(zip(*all_files)):
        all_contents.append([content.strip() for content in contents])
    for idx, atc_content in enumerate(zip(*atcs)):
        all_atcs.append(atc_content)
    return np.array(all_contents), np.array(all_atcs)
    pass


def write_regular_data_to_file(_files, _data):
    for point in _data:
        for file, content in zip(_files, point):
            content = content.strip()
            file.write(content + '\n')
    pass


def write_atcs_to_file(_atc_files , _atcs):
    for file, atc in zip(_atc_files, _atcs):
        print(file, len(atc))
        serialize_to_file(atc, file)
    pass


def write_content_to_file(regular_files, atc_file_paths,
                          train_data, train_atcs,
                          valid_data, valid_atcs,
                          test_data, test_atcs):
    
    train_files, valid_files, test_files = regular_files
    train_atc_files, valid_atc_files, test_atc_files = atc_file_paths
    write_regular_data_to_file(train_files, train_data)
    write_regular_data_to_file(valid_files, valid_data)
    write_regular_data_to_file(test_files, test_data)
    train_atcs = np.transpose(np.asarray(train_atcs))
    valid_atcs = np.transpose(np.asarray(valid_atcs))
    test_atcs = np.transpose(np.asarray(test_atcs))
    write_atcs_to_file(train_atc_files, train_atcs)
    write_atcs_to_file(valid_atc_files, valid_atcs)
    write_atcs_to_file(test_atc_files, test_atcs)
    pass


def main():
    dataset = 'BR_ALL'
    data_dir = 'defj_experiment/data/raw/' + dataset
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_files_name = train_dir + '/files.txt'
    valid_files_name = valid_dir + '/files.txt'
    test_files_name = test_dir + '/files.txt'
    all_train_information = extract_file_information(train_files_name)
    all_valid_information = extract_file_information(valid_files_name)
    all_test_information = extract_file_information(test_files_name)
    all_projects = list(all_train_information['projects'].\
        union(all_valid_information['projects']).union(all_test_information['projects']))

    all_regular_files, all_atc_file_paths = create_appropriate_files_in_project_directory(
        train_dir, valid_dir, test_dir, all_projects)
    train_data, train_atcs = read_data_all(train_dir)
    valid_data, valid_atcs = read_data_all(valid_dir)
    test_data, test_atcs = read_data_all(test_dir)
    # print(len(train_atcs[0]), len(valid_atcs), len(test_atcs))
    print(len(train_data), len(valid_data), len(test_data))
    print(max([max(all_train_information['examples_in_project'][project]) for project in all_projects]))
    print(max([max(all_valid_information['examples_in_project'][project]) for project in all_projects]))
    print(max([max(all_test_information['examples_in_project'][project]) for project in all_projects]))
    for project in all_projects:
        train_example_ids = all_train_information['examples_in_project'][project]
        valid_example_ids = all_valid_information['examples_in_project'][project]
        test_example_ids = all_test_information['examples_in_project'][project]
        print(project, len(train_example_ids), len(valid_example_ids), len(test_example_ids))
        write_content_to_file(
            all_regular_files[project], all_atc_file_paths[project],
            train_data[train_example_ids],
            train_atcs[train_example_ids],
            valid_data[valid_example_ids],
            valid_atcs[valid_example_ids],
            test_data[test_example_ids],
            test_atcs[test_example_ids]
        )


if __name__ == '__main__':
    main()
