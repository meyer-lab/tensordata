from setuptools import find_packages, setup

setup(
    name="tensordata",
    version="0.0.7",
    description="A common repository for tensor structured datasets.",
    url="https://github.com/meyer-lab/tensordata",
    license="MIT",
    packages=find_packages(exclude=["doc"]),
    install_requires=["numpy", "tensorly"],
)
