#!/usr/bin/env python

import os
from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))


def package_files(directory, extensions):
    """Walk package directory to make sure we include all relevant files in package."""
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if any([filename.endswith(ext) for ext in extensions]):
                paths.append(os.path.join('..', path, filename))
    return paths


json_yaml_csv_files = package_files('lrccd', ['yaml', 'json', 'csv', 'h5', 'fchk', 'xyz'])

if __name__ == "__main__":
    print(module_dir)
    setup(
        name='LRC-CD',
        version='0.0.0',
        description='LRC-CD model',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github/yingxingcheng/lrccd',
        author='YingXing Cheng',
        author_email='Yingxing.Cheng@ugent.be',
        license='GNU',
        packages=find_packages(),
        package_data={'lrccd': json_yaml_csv_files},
        zip_safe=False,
        install_requires=[
            'horton>=2.1.0',
            'progress>=1.5',
            'numpy<=1.23.0',
        ],
        extras_require={},
        classifiers=['Programming Language :: Python :: 2.7',
                     'Development Status :: 5 - Production/Stable',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                     'Natural Language :: English',
                     'Intended Audience :: Science/Research',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering :: Chemistry',
                     'Topic :: Scientific/Engineering :: Physics',
                     ]
    )
