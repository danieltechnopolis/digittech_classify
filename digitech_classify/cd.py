import os
import sys

def print_tree(start_path, prefix=''):
    entries = sorted(os.listdir(start_path))
    entries_count = len(entries)
    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = '└── ' if i == entries_count - 1 else '├── '
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = '    ' if i == entries_count - 1 else '│   '
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = '.'
    print(root_dir)
    print_tree(root_dir)
