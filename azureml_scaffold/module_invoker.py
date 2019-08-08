"""
Run official module by command line arguments
"""

import sys

from azureml.studio.modulehost.module_host_executor import execute_cli


if __name__ == '__main__':
    execute_cli(sys.argv)
