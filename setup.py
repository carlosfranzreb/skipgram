import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setuptools.setup(
  name="skipgram",
  version="0.0.1",
  author="Carlos Franzreb",
  author_email="carlosfranzreb[at]gmail.com",
  description="PyTorch implementation of skip-gram (also known as word2vec).",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/carlosfranzreb/skipgram",
  project_urls={
      "Bug Tracker": "https://github.com/carlosfranzreb/skipgram/issues",
  },
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
  ],
  package_dir={"": "skipgram"},
  packages=setuptools.find_packages(where="skipgram"),
  python_requires=">=3.6",
)