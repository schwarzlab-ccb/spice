from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='scna-spice',
    version='0.1.1',
    author='Tom L Kaufmann',
    description='SPICE: Selection Patterns In somatic Copy-number Events',
    author_email='tkau93@gmail.com, marina.55kovic@gmail.com, roland.f.schwarz@gmail.com',
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/schwarzlab/spice',
    packages=find_packages(),
    package_data={
        'spice': ['objects/**/*.yaml', 'objects/**/*.tsv', 'objects/**/*.pickle',
                  'reference_loci/**/*.tsv'],
    },
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'spice=spice.cli:main',
        ],
    },
    install_requires=[
        # 'numpy==1.26.4',
        # 'pandas==2.2.3',
        'numpy',
        'pandas',
        'scipy',
        'seaborn>=0.12',
        'tqdm',
        'fire',
        'pyyaml',
        'joblib',
        'ortools==9.8.3296',
        'importlib_resources>=5.0; python_version < "3.9"',
    ],
    extras_require={
        'snakemake': ['snakemake>=7.0'],
        'preprocessing': ['CNSistent'],
    },
    python_requires='>=3.8',
)
