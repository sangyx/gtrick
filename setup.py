import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="gtrick",
    version="0.0.dev1",
    author="sangyx",
    author_email="sangyunxin@gmail.com",
    description="Bag of Tricks for Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sangyx/gtrick",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch"
    ],
    python_requires=">=3",
)