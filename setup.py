from setuptools import setup, find_packages # type: ignore

setup(
    name="py2cpp",
    version="1.6.1",
    description="A Python-to-C++ translator.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    author="typeulli",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "mypy>=1.8.0",
    ],
    entry_points={
        "console_scripts": [
            "py2cpp = py2cpp.cli:main",
        ],
    },
)