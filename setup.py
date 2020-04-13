from setuptools import setup, find_packages

import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
version = '0.0.0.0.0.0' # year.month.day.hour.minute.second
with open(os.path.join(current_folder,'VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='expert_finding',
      version=version,
      description='expert_finding',
      url='https://github.com/brochier/expert_finding',
      author='Robin Brochier',
      author_email='robin.brochier@univ-lyon2.fr',
      license='MIT',
      include_package_data=True,
      packages=find_packages(exclude=['tests*']),
      package_data={'': ['expert_finding/resources/*']},
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'unidecode',
          'scikit-learn',
          'tensorflow>=1,<2'
      ],
      zip_safe=False)
