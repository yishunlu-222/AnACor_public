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
             'anacor.main = AnACor.main:main',
             'anacor.postprocess = AnACor.postprocess:main',
             'anacor.preprocess_lite = AnACor.preprocess_lite:main',
             'anacor.main_lite = AnACor.main_lite:main',
             'anacor.mp_lite = AnACor.mp_lite:main',
             'anacor.postprocess_lite = AnACor.postprocess_lite:main',],
    },

)
            #'dials @ https://github.com/yishunlu-222/dials_precalcu/tree/precalc_abs_model',
#                install_requires=[
#        'importlib-metadata; python_version == "3.8"',
#        'opencv-python',
#        'scikit-image',
#        'numba',
#        'imagecodecs',
#
#
#    ],