Machine learning algorithms for shared autonomy with brain-computer interfaces.

TODO
----
 - Convert notebooks into scripts
 - Modify TAMER to learn a reward model that generates dense rewards from BCI features

Usage
-----

Install Python dependencies with the [pip](https://pip.pypa.io/en/stable/installing/) package
manager using

```
pip install -r requirements.txt
```

Install [baselines](https://github.com/openai/baselines/tree/cfa1236d786b120fc9e9141c4bc18de3b0c95d13) and replace `baselines/baselines/deepq/simple.py` with `deps/simple.py`.

Open `cursor-control.ipynb` and `typing.ipynb` with [Jupyter](https://jupyter.org/).

Questions and comments
----------------------

Please contact the author at `` if you have questions or find bugs.

Citation
--------
If you find this software useful in your work, we kindly request that you cite the following [paper](https://arxiv.org/abs/):

```
@InProceedings{,
  title={},
  author={},
  booktitle={Arxiv },
  year={},
  url={https://arxiv.org/abs/}
}
```
