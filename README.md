# Covid-19 Model 

In this repository, you'll find the codes used to simulate the COVID-19 model in "A model for COVID-19 with
isolation, quarantine and testing as control measures"
[https://arxiv.org/abs/2005.07661](https://arxiv.org/abs/2005.07661), by M. S.
Aronna, R. Guglielmi and L. M. Moschen.

## The Model 

### Abstract of the article

 In this article we propose a compartmental model for the dynamics of
 Coronavirus Disease 2019(COVID-19).  We take into account the presence of
 asymptomatic infections and the main policiesthat  have  been  adopted  so
 far  for  the  combat  of  this  disease:  isolation  (or  social
 distancing)  ofa portion of the population,  quarantine for confirmed cases
 and testing.  We model isolation byseparating the population in two groups:
 one composed by key-workers that keep working duringthe  pandemic  and  have
 a  usual  contact  rate,  and  a  second  group  consisting  of  people  that
 areenforced/recommended to stay at home.  We refer to quarantine as strict
 isolation, and it is appliedto confirmed infected cases.
 
 In the proposed model, the proportion of people in isolation, the level of
 contact reduction andthe testing rate are control parameters that can vary in
 time, representing policies that evolve indifferent stages.  We obtain an
 explicit expression for the basic reproduction numberR0in terms ofthe
 parameters of the disease and of the control policies.  In this way we can
 quantify the effect thatisolation and testing have in the evolution of the
 epidemic.  We present a series of simulations toillustrate different
 realistic scenarios.  From the expression ofR0and the simulations we
 concludethat isolation (social distancing) and testing among asymptomatic
 cases are fundamental actions tocontrol the epidemic, and the stricter these
 measures are and the sooner they are implemented, themore lives can be saved.
 Additionally, we show that people that remain in isolation
 significantlyreduce  their  probability  of  contagion,  so  risk  groups
 should  be  recommended  to  maintain  a  lowcontact rate during the course
 of the epidemic. 

### The graphic from model 

![Image from the model](images/model.svg)

### System of equations 

![Image from the system](images/equation.svg)

## The Repository Structure

This repository is organized as follows: 

```bash
├── data
│   ├── parameters.yaml
│   ├── r0.txt
│   ├── table_values.txt
│   └── variables
├── images
├── notebooks
├── pyscripts
│   ├── dynamics_model.py
│   ├── execute_model.py
│   ├── __init__.py
│   └── scenarios.py
├── README.md
└── requirements.txt
```

The data folder contains the files needed to organize the experiments: the
parameters used, the r0 retrivied from the article and the table values from
the scenarios described in the section 4 of the article. For each experiment,
one can save the result in the variables folder (it is done automatically). 

The pyscripts folder has the three main files: the ```dynamics_model.py```
file contains the dynamics of the model described as Python functions. The
```execute_model.py``` is how one can experiment the model with different
parameters and ```scenarios.py``` is a file to reproduce the results found in
the paper. 

We expect it is the most reproducible that it can be. In order to experiment,
you need: 

- Have Python 3 installed. You can check this
  [here](https://www.python.org/downloads/). 
- Clone this repository in you machine with the command ```git clone https://github.com/lucasmoschen/covid-19-model``` 
- Install the requirements with ```pip install -r requirements.txt```

## Experimenting 

After the steps above, so as to experiment the model, it's needed to follow
these steps: 

1. Change the parameters in the `parameters.yaml`. Maintain the format of the
   file as already is. 
2. Change the variable functions in the class `Parameters_Functions` in
   `dynamics_model.py` file. 
3. Enter in the `pyscripts` folder and run `python execute_model.py`. Follow
   the instructions to save the variables in the terminal. 

If you want to reproduce the results in the article or change the values and
see how it affects each scenario, it's necessary to follow these steps: 

1. Access the `scenarios.py` file and change the parameters values in your
   way. 
2. Enter in the `pyscripts` folder and run `python scenarios.py`. The result
   will be in `data/table_values.txt`. 

### Sugestions

Please, any sugestions make an issue and I will answer as quick as I can. Thanks!

