# Reproducibility in Tensorflow/PyTorch/JAX

This is an example repository from my blog on [Reproducibility in Tensorflow/PyTorch/JAX](https://wolodjaz.github.io/blogs/), so please read it first.

The structure of this repository differs from the one in the blog due to the addition of [Binder](https://mybinder.readthedocs.io/en/latest/) settings. The repository structure is as follows:
```text
reproduc-ml-tutorial/
  workspace/          # Default location for data sets, logs, models, parameter files.
    train.yaml        #   Train hyper-parameters.
  .dockerignore       # Docker ignore file that prevents workspace directory to be sent to docker server.
  DockerBuildfile     # Docker recipe.
  environment.yml     # Conda environment config file for mybinder
  index.ipynb         # Example notebook from Reproducibility in Tensorflow/PyTorch/JAX part 2
  mlcube.yaml         # MLCube definition file.
  train_jax.py        # Python source code training simple neural network using MNIST data set with JAX.
  train_pytorch.py    # Python source code training simple neural network using MNIST data set with PyTorch.
  train_tensorflow.py # Python source code training simple neural network using MNIST data set with Tensorflow.
  requirements.txt    # Python project dependencies.
  run.sh              # Main bash script that lunches python script based on passed argument
  Singularity.recipe  # Singularity recipe.
```

## Running the "Reproducibility in Tensorflow/PyTorch/JAX Part 2/2"  Notebook

To run the notebook, you can pull this repository and launch `index.ipynb` locally, but you can also click on the badge below to test running it on BinderHub without pulling the repository 😎:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WolodjaZ/reproduc-ml-tutorial/HEAD?labpath=index.ipynb)

## Main Project

In addition to running the notebook, you can also run the main application where you can train MNIST datasets on a basic neural network made in Pytorch/Jax/Tensorflow. You will build a docker image or a singularity image and launch it to run the training. Everything, including logs and data, will be saved under the `workspace` directory. There is also a `train.yaml` file where I have defined all the parameters used for the scripts. You can check and change them if you want to.

## Running the Project

To run the project, we are using [MLCube](https://github.com/mlcommons/mlcube), which provides the contract for our pipeline, as defined in the file `mlcube.yaml`. Based on this file and the framework, you will need to first configure our environment by building our images. Before doing so, please install mlcube:
```bash
pip install mlcube
``` 
Next, create our images:
```bash
# Prepare docker image
mlcube configure --mlcube=. --platform=docker
# Prepare singularity image
mlcube configure --mlcube=. --platform=singularity
```
You can now run our pipelines by choosing which platform and framework to use:
```bash
# Docker
mlcube run --mlcube=. --platform=docker --task=pytorch
mlcube run --mlcube=. --platform=docker --task=tensorflow
mlcube run --mlcube=. --platform=docker --task=jax

# Singularity
mlcube run --mlcube=. --platform=singularity --task=pytorch
mlcube run --mlcube=. --platform=singularity --task=tensorflow
mlcube run --mlcube=. --platform=singularity --task=jax
```
After running the commands, pipeline will start the training process and the log and models will be saved under the `workspace` folder.

## Additional Resources:
- Check out my blog [Reproducibility in Tensorflow/PyTorch/JAX](https://wolodjaz.github.io/blogs/) for more information on the topic.
- [Binder](https://mybinder.readthedocs.io/en/latest/) is a great tool for creating and sharing custom computing environments with others.
- [MLCube](https://github.com/mlcommons/mlcube)  is a useful tool that provides a consistent interface for machine learning models in containers like Docker.
- For more guidance on reproducible research, check out [The Turing Way](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html)

## And Grand Finally
Closing comment offered by chatGPT 🤖

We're so glad you've given our project a try! Your feedback is incredibly valuable to us as we continue to improve and update the project. Whether you have questions, comments, or suggestions, please don't hesitate to reach out to us by emailing us at vladimirzaigrajew@gmail.com or by opening an issue on the GitHub repository. Thank you for your support!