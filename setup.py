from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='consensus-module',
   version='1.0',
   description='A Python module to generate a consensual explanation in AI domains (XAI).',
   license='GNU',
   long_description=long_description,
   author='Luana de Queiroz Garcia',
   author_email='luanaqg@estudante.ufscar.br',
   packages=['consensus_module'],  #same as name
   install_requires=['anchor_exp==0.0.2.0',
    'img2pdf==0.5.1',
    'ipython==8.27.0',
    'joblib==1.4.2',
    'kaleido==0.2.1',
    'lime==0.2.0.1',
    'matplotlib==3.9.2',
    'numpy==1.26.4',
    'pandas==2.2.2',
    'plotly==5.24.1',
    'scikit-learn==1.5.2',
    'shap==0.46.0'
    ], #external packages as dependencies
)