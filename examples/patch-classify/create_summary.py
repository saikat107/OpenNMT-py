p_file = 'Correctly-predicted-patch-codit.txt'
type_stat = {}
with open(p_file) as f:
    for ln, line in enumerate(f):
        if ln%7 == 4:
            line = line.strip()
            change_type = line.split(':')[1].strip()
            if change_type not in type_stat.keys():
                type_stat[change_type] = 0
            type_stat[change_type] += 1

for key in type_stat:
    print(key, ' , ', type_stat[key])
