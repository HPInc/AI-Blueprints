---
title:  'Base Images'
sidebar_position: 6
---
# AI Studio Workspace Images

Workspaces are one of the core elements of AI Studio. Each workspace runs in a separate container, isolated from the host application except when sharing project files and folders. Workspaces in AI Studio leverage IPython notebooks or Python scripts through JupyterLab to help users develop their experiments in a centralized, secure, and efficient way.

## Base Images

Base Images are predefined environments with a pre-installed Python version and a set of libraries designed to help you dive into AI Studio right away. AI Studio offers two different base images to accommodate the scale of your workspaces: one that's GPU enabled and one that's CPU-only. 

CPU-only images are smaller, faster to load, and are most useful when performing basic computations and traditional DS algorithms. CPU-only images have most of the same libraries as the more robust GPU-enabled options, so users that don't have a GPU that's compatible with AI Studio are still likely to find them helpful. GPU enabled images support cuda packages and attach notebooks to the device to run on GPUs without additional configuration by the user. 


### Minimal Image

The CPU-only minimal image comes with the minimal set of libraries necessary to run Data Science experiments in AI Studio. It runs JupyterLab 4.0.1 with Python 3.10.11, based on the [minimal-notebook:python-3.10](https://hub.docker.com/r/jupyter/minimal-notebook/tags) image from Jupyter Docker hub 


### Deep Learning Image: 

The Deep Learning image is specifically tailored for Deep Learning and Neural Network Architectures (NNA). It encompasses an extensive configuration designed to optimally run your most intricate experiments. High computational power is necessary, making the Deep Learning image well-suited for tasks involving image analytics and Large Language Models (LLMs). It includes all the features found in the Data Science image, with upgraded libraries like TensorFlow and PyTorch. The Deep Learning image also comes pre-configured with GPU capabilities for which we recommend reserving at least 4GB of memory. 
  

### Data Science Image: 

The Data Science image represents the most standard yet powerful configuration, designed for quick setup and enhanced performance. Inspired by scipy image on Jupyter Docker hub, this image encompasses all the features of the minimal configuration and further enhances it with Data Visualization tools like Seaborn, Altair, and scikit-image. Recommended for usage with a minimum of 4GB of memory. 

## NGC Catalog Containers

NVIDIA's NGC Catalog includes dozens of containers, which you can leverage in AI Studio with the same steps you’d use to add a base image to a workspace. Any of the NGC containers and models that are available in AI Studio during workspace creation and asset management are supported and available to all users.

:::note
Some of the models and containers listed on [NVIDIA's NGC Catalog](https://www.nvidia.com/en-us/gpu-cloud/) website may not be compatible with your projects in AI Studio.
::: 

Three images we have examples for and that run optimally on AIS include: 

- [NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo): NVIDIA NeMo™ is an end-to-end platform for development of custom generative AI models. 

- [RAPIDS Base](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/base): Execute end-to-end data science and analytics pipelines entirely on GPUs. Use this image if you want to use RAPIDS as a part of your pipeline. Visit [rapids.ai](https://rapids.ai/) for more information. 

- [RAPIDS Notebooks](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/notebooks): Execute end-to-end data science and analytics pipelines entirely on GPUs. Use this image if you want to explore RAPIDS through notebooks and examples. Visit [rapids.ai](https://rapids.ai/) for more information. 

Check out our guide on [Using a Predefined Workspace](/docs/aistudio/using-aistudio/workspaces/predefined-workspace.md) for more details on how to add an NGC container like the ones above to your AI Studio workspace. 

:::tip
When you’re creating a workspace or adding an asset in AI Studio, you can double-click on an NGC container’s name to open NVIDIA’s documentation about the container in your web browser.
:::  


#### The following libraries come pre-installed: 

:::note

As newer versions of these libraries are released, the ones your workspace leverages might become outdated. Use ```pip list --outdated``` to view a list of your outdated packages.

:::

|Name|Description|Minimal|Deep Learning|Data Science|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|altair|A declarative statistical visualization library for Python with a simple, friendly, and consistent API. Built on top of the powerful Vega-Lite JSON specification.|❌|✔|✔|
|bokeh|An interactive visualization library for modern web browsers. It provides elegant and streamlined construction of versatile graphics and offers high-performance interactivity across large or streaming asset.|❌|✔|✔|
|ipympl|A Jupyter extension that improves chart visualization in matplotlib.|❌|✔|✔|
|matplotlib|A comprehensive library for creating static, animated, and interactive visualizations in Python.|✔|✔|✔|
|mlflow|A platform used to streamline ML development. It lets users log and track ML experiments; pack and register ML models; and version ML projects, among other useful features.|✔|✔|✔|
|numpy|A fundamental package for scientific computing with Python. It provides a powerful N-dimensional array object; sophisticated broadcasting functions; and tools for seamless C/C++ and Fortran code integration. It also offers useful tools for linear algebra, Fourier transform, and random number capabilities.|✔|✔|✔|
|pandas|A package that provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive.|✔|✔|✔|
|patsy|A library for describing statistical models (especially linear models, or models that have a linear component).|✔|✔|✔|
|pillow|A Python Imaging Library fork that adds image processing capabilities to the Python interpreter.|✔|✔|✔|
|plotly-express| A higher-level wrapper to the plotly data visualization library. It enables interactive data visualizations.|❌|✔|✔|
|pytorch|A widely-used Deep Learning library.|❌|✔|❌|
|scikit-image|An open-source image processing library for the Python programming language. It includes algorithms for segmentation, geometric transformations, color space manipulation, analysis, filtering, morphology, feature detection, and more.|❌|✔|✔|
|scikit-learn|A Python module for machine learning built on top of SciPy. It provides a wide range of algorithms for classification, regression, clustering, feature extraction, anomaly detection and others.|✔|✔|✔|
|scipy|An open-source library for mathematics, science, and engineering. It builds on NumPy and provides additional functionality for optimization, integration, interpolation, eigenvalue problems, signal and image processing, statistical functions, and more.|✔|✔|✔|
|statsmodel|A Python library that provides classes and functions for estimating statistical models, conducting statistical tests, and exploring data.|✔|✔|✔|
|tensorboard|A library that logs information through ML model training. It also adds an interface for visualizing and understanding the evolution of logged training.|✔|✔|✔|
|tensorflow|A widely-used Deep Learning library.|❌|✔|❌|
|tqdm|A library that displays progress bars in loops and iterations in Python. It lets users track the progress of lengthier tasks visually.|❌|✔|✔|
|zstandard|A fast and efficient compression library. It offers competitive compression rates at significantly higher speed when compared to standard compression libraries.|✔|✔|✔|

:::note

You can run ```pip install [package-name] -U``` to update these libraries when necessary.

:::
