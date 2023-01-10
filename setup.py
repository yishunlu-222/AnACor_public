from setuptools import setup

setup(
    name='AnACor',
    version='1.0',
    description='AnACor for analytical absorption correction by tomography reconstruction',
    author='Yishun Lu',
    author_email='yishun.lu@eng.ox.ac.uk, wes.armour@oerc.ox.ac.uk',


    entry_points = {
        'console_scripts' :
            ['anacor.preprocess = AnACor.preprocess:main',
             'anacor.main = AnACor.main:main',],
    }
)
#    install_requires=[
#        'importlib-metadata; python_version == "3.8"',
#        'opencv-python',
#        'scikit-image',
#        'numba',
#        'imagecodecs',
#        'dials @ https://github.com/yishunlu-222/dials_precalcu/tree/precalc_abs_model',
#
#    ],