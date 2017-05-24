#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "This package enable the analysis of 4D tissue data using spatio-temporal cellular features extraction, organisation and clustering."
readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')


# find version number in src/vplants/tissue_analysis/version.py
version = {}
with open("src/vplants/tissue_analysis/version.py") as fp:
    exec(fp.read(), version)


setup_kwds = dict(
    name='vplants.tissue_analysis',
    version=version["__version__"],
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Jonathan Legrand, Frederic Boudon, ",
    author_email="jonathan.legrand AT ens-lyon.fr, frederic.boudon AT cirad.fr, ",
    url='https://github.com/Jonathan Legrand/tissue_analysis',
    license='cecill-c',
    zip_safe=False,

    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        ],
    tests_require=[
        "mock",
        "nose",
        "sphinx",
        ],
    entry_points={},
    keywords='',
    test_suite='nose.collector',
)
# #}
# change setup_kwds below before the next pkglts tag

setup_kwds['entry_points']['oalab.world'] = ['oalab.world/tissue_analysis = vplants.tissue_analysis.tissue_analysis_oalab.plugin.world']

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}
