from setuptools import setup, find_packages

setup(
    name="quick-eda",
    version="0.3.0",
    author="Tarunpreet Singh",
    description="TL;DR for your dataset — quick insights, minimal noise, actionable results.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3",
        "numpy>=1.21",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "IPython",
        "jinja2>=3.0",
        "xlrd>=2.0",
        "openpyxl>=3.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.8",
)