from json import load


def ipynb_to_py(filepath: str) -> str:
    with open(filepath) as f:
        data = load(f)
    src = ''
    for index, cell in enumerate(data['cells'], 1):
        if cell['cell_type'] != 'code':
            continue
        src += f'# Cell {index}\n'
        src += ''.join(
            map(lambda x: f'# {x}' if x.startswith('%') else x, cell['source'])
        )
        src += '\n\n'

    new_file = filepath.replace('.ipynb', '.py')
    with open(new_file, 'w') as f:
        f.write(src)
    print(f'Created file {new_file}')


if __name__ == '__main__':
    import os
    # ignore_dirs = ['.git', '.ipynb_checkpoints']
    # ignore_files = ['.gitignore', 'cleanup.py', 'ipynb_to_py.py']
    # filenames = []
    # ROOT = '.'
    # for root, dirs, files in os.walk(ROOT, topdown=True):
    #     dirs[:] = list(filter(lambda x: x not in ignore_dirs, dirs))
    #     files[:] = list(filter(lambda x: x not in ignore_files and x.lower().endswith('.ipynb'), files))
    #     filenames += list(map(lambda x: os.path.join(root, x), files))

    filenames = ['HW2/hw_PCA.ipynb']
    for filename in filenames:
        try:
            ipynb_to_py(filename)
        except Exception as e:
            print(f'Error Occured Converting {filename}!')
            print(e)
