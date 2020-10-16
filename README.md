# SEIR-Campus

**Citation info**: Zalesak, M; Samaranayake, S.  2020.  "[SEIR-Campus: Modeling Infectious Diseases on Univeristy Campuses](SeirCampus.pdf)." 

SEIR-Campus is a Python package designed to aid the modeling and study of infectious diseases spreading in communities with a focus on fast computations for large university campuses. It uses an agent based framework with interactions derived from individual movement patterns and interactions. For example, in the university setting using course registration data and models of student social dynamics to simulate day-by-day spread of infections in discrete time.  Its features include:

* An epidemiological model based on the SEIR model
* Distinction between symptomatic and asymptomatic cases
* Modeling interactions (e.g., course contacts and social interactions) that change day-by-day
* Specifying stochastic model parameters (e.g., the infectious period, recovery time etc. can be a stochastic process) 
* Highly customized testing protocols and quarantine procedures for individuals
* Integration of contact tracing
* Ability to add custom social networks
* Fast computation of large problem instances (e.g., 20,000 individuals for a semester in less than five seconds)

This repository contains the Python package PySeirCampus, located in the folder PySeirCampus.  Demonstrations of how to use the package are contained in the Jupyter notebook file [Examples.ipynb](Examples.ipynb).  A full description of the SEIR-Campus model, as well as explanations to go along with the examples, can be found in [our paper](SeirCampus.pdf).

In order to use this package, you will need to have the following installed on your system:

* Python 3
* Numpy
* Matplotlib
* Pandas
* Jupyter Notebook

To being using PySeirCampus, place the PySeirCampus folder in your working directory.  From Python 3, import the package using

```
import PySeirCampus
```

