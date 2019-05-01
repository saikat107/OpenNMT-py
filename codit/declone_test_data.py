import os
import pickle

base_dir = '/home/saikatc/Research/OpenNMT-py/icse_data/raw/all/concrete_small'
test_dir = base_dir + '/test_new'
tmp_file = '/home/saikatc/Research/OpenNMT-py/tmp/icse-all-concrete-small'

prev_token_file = open(test_dir + '/prev.token')
next_token_file = open(test_dir + '/next.token')
examples = []
taken_indices = []

for idx, (pc, cc) in enumerate(zip(prev_token_file, next_token_file)):
    pcs = pc.strip()
    ccs = cc.strip()
    key = pc + cc
    if key in examples:
        continue
    else:
        examples.append(key)
        taken_indices.append(idx)
prev_token_file.close()
next_token_file.close()

print(len(taken_indices))

files_names = ['next.augmented.rule',
                'next.augmented.token',
                'next.frontier',
                'next.parent.rule',
                'next.parent.time',
                'next.rule',
                'next.token',
                'next.token.id',
                'prev.augmented.rule',
                'prev.augmented.token',
                'prev.frontier',
                'prev.parent.rule',
                'prev.parent.time',
                'prev.rule',
                'prev.token',
                'prev.token.id']

if not os.path.exists(base_dir + '/test-declone/'):
    os.mkdir(base_dir + '/test-declone/')
for file_name in files_names:
    new_version_file_name = base_dir + '/test-declone/' + file_name
    new_version_file = open(new_version_file_name, 'w')
    old_version_file = open(test_dir + '/' + file_name)
    for idx, line in enumerate(old_version_file):
        line = line.strip()
        if idx in taken_indices:
            new_version_file.write(line + '\n')
    old_version_file.close()
    new_version_file.close()

tmp_file_decloned = tmp_file + '-decloned'
with open(tmp_file) as told:
    with open(tmp_file_decloned, 'w') as tnew:
        for idx, line in enumerate(told):
            line = line.strip()
            if idx in taken_indices:
                tnew.write(line + '\n')
        tnew.close()
    told.close()

atc_files = ['atc_method.bin', 'atc_scope.bin']
for atc_file in atc_files:
    atold = test_dir + '/' + atc_file
    f = open(atold , 'rb')
    old_atcs = pickle.load(f)
    f.close()
    new_atcs = []
    for idx in taken_indices:
        new_atcs.append(old_atcs[idx])
    atnew = base_dir + '/test-declone/' + atc_file
    fn = open(atnew, 'wb')
    pickle.dump(new_atcs, fn)
    fn.close()
