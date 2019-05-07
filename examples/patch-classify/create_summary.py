p_file = 'result/'
t2t = p_file + 'only-t2t-category.txt'
s2s = p_file + 'only-s2s-category.txt'
common = p_file + 'common-category.txt'

files = [t2t, s2s, common]
type_stat_from_baishakhi = [{}, {},{}]
type_stat_misc = [{},{},{}]

for tidx, p_file in enumerate(files):
    with open(p_file) as f:
        eid = ''
        for ln, line in enumerate(f):
            if ln % 7 == 0:
                eid = line.strip()
            if ln % 7 == 4:
                line = line.strip()
                change_type = line.split(':')[1].strip()
                if change_type == '':
                    print(p_file, eid, line)
                if '/' in change_type:
                    change_type_b = change_type.split('/')[1].strip()
                    if change_type_b == '':
                        print(p_file, eid, line)
                    change_type_m = 'Misc'
                else:
                    change_type_b = change_type.strip()
                    change_type_m = change_type_b

                if change_type_b not in type_stat_from_baishakhi[tidx].keys():
                    type_stat_from_baishakhi[tidx][change_type_b] = 0
                type_stat_from_baishakhi[tidx][change_type_b] += 1

                if change_type_m not in type_stat_misc[tidx].keys():
                    type_stat_misc[tidx][change_type_m] = 0
                type_stat_misc[tidx][change_type_m] += 1
        f.close()

for tidx, p_file in enumerate(files):
    b_stat = type_stat_from_baishakhi[tidx]
    print('=================================================================')
    print(p_file)
    for key in b_stat.keys():
        print(key, b_stat[key], sep='\t')
    print('=================================================================')
