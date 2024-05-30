import shutil


def read_test_file(path):
    test_names = []
    with open(path, 'r') as f:
        test_pairs = f.read().split('\n')
    for pair in test_pairs:
        pair = pair.split('\t')
        if len(pair) == 3:
            test_names.append(pair[0])
        elif len(pair) == 4:
            test_names.append(pair[0])
            test_names.append(pair[2])
    return test_names


def change_test_dir(test_names, path_from, path_to):
    for folder in test_names:
        try:
            shutil.move(f'{path_from}/{folder}', path_to)
            shutil.rmtree(f'{path_from}/{folder}')
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    path_from = 'path/to/all/data'
    path_to = 'path/to/place/test/data'
    test_file = 'path/to/test/file'
