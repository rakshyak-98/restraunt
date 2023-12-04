from setuptools import setup, find_packages

setup(
    name='RestRec',
    version='0.0.0.0',
    packages=find_packages(),
    install_requires = ['flask_sqlalchemy', 'bcrypt', 'numpy', 'pandas', 'seaborn', 'scikit-learn'],
    url='',
    license='MIT',
    author='Arnold Anthony',
    author_email='',
    description='A restaurant recommendation system'
)
