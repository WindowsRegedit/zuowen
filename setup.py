import os
import shutil
from setuptools import setup, find_packages

__version__ = "2022.6.29r2"

# Thanks to
# https://stackoverflow.com/questions/72513435/how-can-i-create-my-setup-py-with-non-python-files-and-no-python-files-folders
def safe_del(folder):
    CUR_PATH = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(CUR_PATH, folder)
    if os.path.isdir(path):
        print("del dir ", path)
        shutil.rmtree(path)


safe_del("build")
safe_del("dist")
safe_del("zuowen.egg-info")

with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = f.read()
    requires = requires.split("\n")

setup(
    name="zuowen",
    author_email="admin@wufan.fun",
    author="Fan Wu",
    description="Python 作文生成器",
    keywords="UI CLI Torch Train Transformers Jieba SentencePiece",
    url="https://wufan.fun/",
    project_urls={
        "Source Code": "https://github.com/WindowsRegedit/zuowen",
    },
    version=__version__,
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "zuowen-ui = zuowen.run_ui:main",
            "zuowen = zuowen.generate:main",
            "zuowen-preprocess = zuowen.preprocess:main",
            "zuowen-trainer = zuowen.train:main"
        ]
    },
    install_requires=requires,
    tests_require=[
        'pytest>=3.3.1',
        'pytest-cov>=2.5.1',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # 'Programming Language :: Python :: 3.10',
        # 'Programming Language :: Python :: 3.11',
    ],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={'zuowen': ['config/*', "vocab/*", "ui/markdown/*", "ui/pages/*"]},
)
