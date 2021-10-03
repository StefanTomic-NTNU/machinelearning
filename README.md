# K-means and Logistic Regression 

To see the algorithms in action, simply open the `.ipynb` (jupyter notebook) files. 
This will display the results of the algorithms, even in Github. 

This project was completed as the individual assignment in the subject _tdt4173 - Machine learning_

## Problem Description

In this assignment you will be developing well-known and simple (but occasionally very useful) machine learning algorithms. The interface you are asked to implement strongly resembles the one used in [Scikit Learn]( https://scikit-learn.org/stable/). However, you are not allowed to use this library (or any other third-party solutions). Instead, you will implement the algorithms from scratch, using only basic tools like numpy.

### Overview
- You are presented with three common machine learning algorithms; K-Means, Decision Tree, and Logistic Regression.
- All material for each algorithm can be found in its own self-contained folder.
- The <algorithm_name>.py files (e.g. k_means.py) contains skeleton code for an sklearn-style implementation.
- For each algorithm, there are two datasets that you are asked to solve. 
  - The data_1.csv files contain very easy problems that can be used to debug your implementation of the algorithm.
  - The data_2.csv files contain slightly harder problems that require you to tweak your machine learning pipeline for good results.
- The experiment.ipynb files are jupyter notebooks with additional instructions/tips, as well as code for loading the datasets, training, and evaluating the models. You should be able to run them once the algorithm files are implemented. They also contain reference values for decent performance on all the problems. 

## Setup 

Python >= 3.6 is required. Pipenv, as well as jupyter lab should be installed.

Dependences are in [requirements.txt](requirements.txt). 
To install them run: `pipenv install -r requirements.txt`


### Running a Notebook Server

After you have installed all the dependencies, you can run a notebook server with:

```
jupyter lab
```

This will start a slightly fancier version of [jupyter notebook](https://jupyter.org). It is a single-page application that allows you to navigate, edit, and run python and jupyter notebook files from your browser. By default the server should be exposed at `localhost` port `8888`. If you're running this command from your laptop or desktop computer, it should automatically open in your default web browser. If for some reason not, try manually navigating to `localhost:8888` in your web browser (or copy the full URL from the logging output in the shell you ran the command). If it asks for a password or token, this can also be found in the shell output.

From here, you can start running and editing the files in the project. If you need more help with the interface, there are several [guides online](https://www.youtube.com/watch?v=7wfPqAyYADY).



