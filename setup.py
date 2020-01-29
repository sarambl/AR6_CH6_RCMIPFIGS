from setuptools import setup

setup(
    name='AR6_CH6_RCMIPFIGS',
    version='v00',
    packages=['ar6_ch6_rcmipfigs', 'ar6_ch6_rcmipfigs.data_in', 'ar6_ch6_rcmipfigs.notebooks',
              'ar6_ch6_rcmipfigs.data_postproc'],
    url='https://github.com/sarambl/AR6_CH6_RCMIPFIGS.git',
    license='MIT',
    author='sarambl',
    author_email='s.m.blichner@geo.uio.no',
    description='', install_requires=['xarray', 'pandas', 'pyam', 'scmdata', 'numpy', 'seaborn', 'matplotlib', 'tqdm',
                                      'cftime', 'IPython']
)
