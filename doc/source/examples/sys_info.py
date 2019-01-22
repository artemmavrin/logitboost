"""Get information about the current platform, Python, and various packages."""

from __future__ import print_function


def _info():
    """Print information about the current machine, Python, and scientific
    computing packages.
    """
    import collections
    import importlib
    import platform
    import sys

    pkgs = ['numpy', 'scipy', 'matplotlib', 'seaborn', 'sklearn']

    def pkg_ver(name):
        try:
            module = importlib.import_module(name)
        except ImportError:
            return None

        version = getattr(module, '__version__', '(version unknown)')
        return version

    # Python dicts are guaranteed to remember insertion order only as of Python
    # 3.7 (and in CPython as an implementation detail as of Python 3.6). Because
    # of this, we use OrderedDict here
    # https://docs.python.org/3.7/library/stdtypes.html#typesmapping

    mach_info = collections.OrderedDict([
        ('Platform', platform.platform()),
        ('Machine Type', platform.machine()),
        ('Processor', platform.processor()),
    ])

    py_info = collections.OrderedDict([
        ('Version', sys.version),
        ('Implementation', platform.python_implementation()),
    ])

    pkg_info = collections.OrderedDict(zip(pkgs, map(pkg_ver, pkgs)))

    all_keys = set().union(mach_info.keys(), py_info.keys(), pkg_info.keys())
    padding = max(map(len, all_keys))
    separator = ': '

    def pprint_info(label, mapping):
        print('', label, '=' * len(label), sep='\n')
        for k, v in mapping.items():
            if not isinstance(v, str):
                v = str(v)
            if '\n' in v:
                v = v.replace('\n', '\n' + ' ' * (padding + len(separator)))
            print(k.rjust(padding), separator, v, sep='')

    pprint_info('Machine', mach_info)
    pprint_info('Python', py_info)
    pprint_info('Packages', pkg_info)


_info()

# Disable importing from this module
del _info
