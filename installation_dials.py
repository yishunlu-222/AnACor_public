"""
install dials developer version
"""
"wget https://raw.githubusercontent.com/dials/dials/main/installer/bootstrap.py"
"python bootstrap.py"
"source dials"
""" No module named  gltbx_gl_ext   https://github.com/dials/dials/issues/1465#issuecomment-715457232 """


""" install AnACor """
"pip install --upgrade build"
"python -m build"
# then you can find the wheel file in the dist folder
"e.g. pip install dist/AnACor-1.2-py3-none-any.whl"