import sys, os


def read_file(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            idx = int(parts[0])
            dist = int(parts[1])
            data[idx] = [idx, dist]
        f.close()
    return data


def write_data(data, path):
    with open(path, 'w') as f:
        for row in data:
            f.write(','.join([str(i) for i in row]) + '\n')
        f.close()

if __name__ == '__main__':
    data_type = 'icse' # sys.argv[1]
    s2s_correct_file = data_type + '-s2s-correct-ids.csv'
    s2s_data = read_file(s2s_correct_file)
    t2t_correct_file = data_type + '-t2t-correct-ids.csv'
    t2t_data = read_file(t2t_correct_file)
    s2s_ids = set(s2s_data.keys())
    t2t_ids = set(t2t_data.keys())
    s2s_only = s2s_ids.difference(t2t_ids)
    common = s2s_ids.intersection(t2t_ids)
    t2t_only = t2t_ids.difference(s2s_ids)
    print(len(s2s_only))
    print(len(t2t_only))
    print(len(common))
    s2s_only_data = []
    t2t_only_data = []
    common_data = []
    all_ids = s2s_ids.union(t2t_ids)

    for idx in all_ids:
        if idx in s2s_only:
            s2s_only_data.append(s2s_data[idx])
        elif idx in t2t_only:
            t2t_only_data.append(t2t_data[idx])
        else:
            common_data.append(t2t_data[idx])

    print(len(s2s_only_data))
    print(len(t2t_only_data))
    print(len(common_data))
    write_data(s2s_only_data, data_type + '/only-s2s.csv')
    write_data(t2t_only_data, data_type + '/only-t2t.csv')
    write_data(common_data, data_type + '/common.csv')

