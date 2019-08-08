import shlex
import subprocess
import sys


def run(command: str, timeout=60000):
    if not command:
        return

    print("\n$ {}".format(command))
    return subprocess.Popen(shlex.split(command), stdout=sys.stdout, stderr=sys.stderr).wait(timeout=timeout)


INITIAL_COMMANDS = '''
pwd
'''.splitlines()

INVOKER_VERSION = '0.0.2'


def is_invoking_official_module(args):
    """
    If invoking official module

    :param args: list
    :return: bool
    """
    return len(args) >= 3 and args[0] == 'python' and args[1] == '-m' and args[2].startswith('azureml.studio.')


if __name__ == '__main__':
    args = sys.argv[1:]
    is_custom_module = not is_invoking_official_module(args)
    module_type = 'custom module' if is_custom_module else 'official module'
    print('Invoking {} by invoker {}.'.format(module_type, INVOKER_VERSION))
    print('')
    print('args: ({} items)'.format(len(args)))
    print('--------------------------------------------')
    for arg in args:
        print(arg)
    print('--------------------------------------------')

    for command in INITIAL_COMMANDS:
        run(command)

    if not is_custom_module:
        index = args.index('-m')
        args[index:index + 2] = ('module_invoker.py',)

    ret = run(' '.join(args))
    exit(ret)
