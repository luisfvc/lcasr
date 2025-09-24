import setuptools

# python setup.py develop --user
setuptools.setup(
    name='lcasr',
    version='0.1dev',
    description='Long-context audio-sheet music retrieval',
    packages=setuptools.find_packages(),
    author='Luis Carvalho',
)
