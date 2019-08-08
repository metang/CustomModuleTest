from setuptools import setup, find_packages


setup(
    name="Movie Lens Data",
    version="0.0.1",
    description="Movie Lens Data for Recommender test",
    packages=find_packages(exclude=["azureml_scaffold"]),
    author="Meng Tang",
    license="MIT",
    include_package_data=True,
)
