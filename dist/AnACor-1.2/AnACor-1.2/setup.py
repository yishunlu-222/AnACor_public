from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
class CustomBuild(build_ext):
    def run(self):
        # run Makefile or other build commands
        subprocess.check_call(['cd', './AnACor/src'])
        subprocess.check_call(['make', 'cpu'])
        super().run()

setup(
    name='AnACor',
    version='1.2',
    packages=find_packages(),
        cmdclass={
        'build_ext': CustomBuild,
    },
    description='AnACor for analytical absorption correction by tomography reconstruction',
    author='Yishun Lu',
    author_email='yishun.lu@eng.ox.ac.uk, wes.armour@oerc.ox.ac.uk',


    entry_points = {
        'console_scripts' :
            ['anacor.preprocess = AnACor.preprocess_lite:main',
             'anacor.main = AnACor.main_lite:main',
            #  'anacor.postprocess = AnACor.postprocess:main',
             'anacor.preprocess_lite = AnACor.preprocess_lite:main',
             'anacor.main_lite = AnACor.main_lite:main',
             'anacor.mp_lite = AnACor.mp_lite:main',
            #  'anacor.postprocess_lite = AnACor.postprocess_lite:main',
             'anacor.init = AnACor.initialization:main' ,
             ],
    },

    install_requires=[
        'importlib-metadata; python_version == "3.8"',
        'opencv-python>=4.6.0',
        'scikit-image>=0.19.3',
        'numba',
        'imagecodecs',


    ],)
            #'dials @ https://github.com/yishunlu-222/dials_precalcu/tree/precalc_abs_model',