if __name__ == '__main__':
    methods_per_bug = dict()
    file_path = "data/raw/BR_ALL/test/files.txt"
    with open(file_path) as f:
        for line in f:
            fn = str(line.strip())
            parts_of_fn = fn.split('/')
            if 'parent' in parts_of_fn:
                pidx = parts_of_fn.index('parent')
                project_name = parts_of_fn[pidx - 2]
                bugid = parts_of_fn[pidx - 1]
                project_bug_id = project_name + " " + bugid
            else:
                project_bug_id = fn
            if project_bug_id not in methods_per_bug.keys():
                methods_per_bug[project_bug_id] = 0
            methods_per_bug[project_bug_id] += 1
    for bugid in methods_per_bug.keys():
        print('\t'.join(bugid.split(" ")), '\t', methods_per_bug[bugid])
    print(len(methods_per_bug.keys()))
