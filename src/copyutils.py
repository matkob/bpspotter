import os
from os import listdir
from shutil import copyfile

if not os.path.exists('data/true'):
    os.mkdir('data/true')
if not os.path.exists('data/false'):
    os.mkdir('data/false')

src_prefix = '/home/mati/CLionProjects/mtdetector/labeled/'

files = listdir(src_prefix)
for file in files:
    if 'false' in file.lower():
        copyfile(src_prefix + file, f'data/false/{file}')
        print(f'{file} false copied')
    elif 'true_1' in file.lower():
        copyfile(src_prefix + file, f'data/true/{file}')
        print(f'{file} true copied')
print('done')
