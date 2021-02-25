Latent Gaussian process with composite likelihoods and numerical quadrature
===========================================================================================
This repository contains the python scripts used in our [paper](https://arxiv.org/abs/1909.01614) published in the Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS) 2021, San Diego, California, USA.

Overview
--------
In this work, we propose an unsupervised generative model that can learn a low-dimensional representation among the observations in a latent space, while making use of all available data in a heterogeneous data setting with missing values. We improve upon the existing Gaussian process latent variable model (GPLVM) by incorporating multiple likelihoods and deep neural network parameterised back-constraints to create a non-linear dimensionality reduction technique for heterogeneous data. In addition, we develop a variational inference method for our model that uses numerical quadrature.<br/><br/><br/>
![Model overview. Each point in the image corresponds to an observation comprising of measurements from different likelihoods, that has been projected on to a two dimensional latent space. The colour coding in the latent space would correspond to cluster membership.](./images/overview_sample.png)
<br/><br/>

Prerequisites
---------------
These scripts require the following software:
- Python (>= 2.6.0)
- [Theano](https://pypi.org/project/Theano/#description)(>= 0.9.0) and associated dependencies

Using our method
----------------
- Create results folder.
- Save the data files in the data folder.
- Update the necessary parameters in the run_model.py file.
- To train the **model**, run:
		`python run_model.py`

Cite
---------------
Please cite this work as:
```
Ramchandran, S., Koskinen, M., & Lähdesmäki, H. (2021). Latent Gaussian process with composite likelihoods and numerical quadrature. Proceedings of the Twenty Fourth International Conference on Artificial Intelligence and Statistics (AISTATS)
```

Authors
---------------
- [Siddharth Ramchandran](https://www.siddharthr.com)
- [Miika Koskinen](https://www.linkedin.com/in/miika-koskinen-462ab83/)
- [Harri Lähdesmäki](http://users.ics.aalto.fi/harrila/)

License
---------------
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
