from setuptools import setup

setup(
    name='AnACor1',
    version='1.1',
    description='AnACor for analytical absorption correction by tomography reconstruction',
    author='Yishun Lu',
    author_email='yishun.lu@eng.ox.ac.uk, wes.armour@oerc.ox.ac.uk',


    entry_points = {
        'console_scripts' :
            ['anacor1.preprocess = AnACor.preprocess:main',
             'anacor1.main = AnACor.main:main',
             'anacor1.postprocess = AnACor.postprocess:main',
             'anacor1.preprocess_lite = AnACor.preprocess_lite:main',
             'anacor1.main_lite = AnACor.main_lite:main',
             'anacor1.mp_lite = AnACor.mp_lite:main',
             'anacor1.postprocess_lite = AnACor.postprocess_lite:main',
             'anacor1.init = AnACor.initialization:main' ,
             ],
    },

    install_requires=[
        'importlib-metadata; python_version == "3.8"',
        'opencv-python',
        'scikit-image',
        'numba',
        'imagecodecs',


    ],)
            #'dials @ https://github.com/yishunlu-222/dials_precalcu/tree/precalc_abs_model',