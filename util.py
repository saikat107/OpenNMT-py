import inspect
import os
from datetime import datetime

def debug(*msg):
    time = datetime.now()
    timestr = str(time.strftime('%X'))
    import inspect
    file_path = inspect.stack()[1][1]
    line_num = inspect.stack()[1][2]
    file_name = file_path
    if os.getcwd() in file_path:
        file_name = file_path[len(os.getcwd())+1:]
    stack = str(file_name) + '#' + str(line_num) + ' [' + timestr + ']'
    print(stack, end=' ')
    res = '\t'
    for ms in msg:
        res += (str(ms) + ' ')
    print(res)


def write_dummy_generated_node_types(input_file, output_file):
    with open(input_file) as inp:
        with open(output_file, 'w') as out:
            for line in inp:
                line = line.strip()
                words = line.split()
                seq_str = []
                for word in words[:-1]:
                    parts = word.split(u"|")
                    seq_str.append(parts[1])
                wstr = ' '.join(seq_str)
                out.write(wstr + '\n')
            out.close()
            inp.close()
    pass


if __name__ == '__main__':
    debug('hello')
