"""Insert dedicated methods into the user's existing main file with a backup.

Run with the learning venv (Python 3.11). No ROS imports or network access.
"""
import argparse
import ast
import shutil
import time
from pathlib import Path


def patched_source(source, methods):
    if 'Kanami: Task2a DQN ROS bridge START' in source:
        raise ValueError('Bridge already installed; do not install twice')
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WrsMainController']
    if len(classes) != 1:
        raise ValueError('Expected one WrsMainController class')
    cls = classes[0]
    runs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    if len(runs) != 1:
        raise ValueError('Expected one run() method')
    run = runs[0]
    executable = list(run.body)
    if (executable and isinstance(executable[0], ast.Expr) and
            isinstance(executable[0].value, ast.Constant) and
            isinstance(executable[0].value.value, str)):
        executable = executable[1:]
    if not executable:
        raise ValueError('run() has no executable body')
    dispatch = ('        # Kanami: opt-in DQN Task2a mode\n'
                '        if rospy.get_param("~task2a_dqn_enabled", False):\n'
                '            self.task2a_dqn_run()\n'
                '            return\n')
    lines = source.splitlines(keepends=True)
    edits = [(cls.end_lineno, '\n'+methods.rstrip()+'\n'),
             (executable[0].lineno-1, dispatch)]
    for index, text in sorted(edits, reverse=True):
        lines.insert(index, text)
    result = ''.join(lines)
    ast.parse(result, feature_version=(3, 6))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    args = parser.parse_args()
    path = Path(args.main).expanduser().resolve()
    if not path.is_file():
        parser.error('Main file not found: '+str(path))
    methods = (Path(__file__).parent/'ros_methods.txt').read_text()
    result = patched_source(path.read_text(), methods)
    backup = path.with_name(path.name+'.before_dqn_'+str(time.time_ns()))
    shutil.copy2(path, backup)
    path.write_text(result)
    print('Backup:', backup)
    print('Updated:', path)
    print('Original mode is retained unless ~task2a_dqn_enabled is true.')
