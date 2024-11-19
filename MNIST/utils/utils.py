import os

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def ensure_dirs(dirs):
    for d in dirs:
        ensure_dir(d)