from setuptools import setup

setup(
    name="crossformer", 
    packages=["crossformer", "cf_scripts"],
    package_dir={
        'crossformer': 'crossformer',  # Maps the package `crossformer` to the directory
        'cf_scripts': 'scripts'        # Maps the package `cf_scripts` to the folder named `scripts`
    },
)
