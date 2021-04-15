# Covid-19 Model 

In this repository, one will find the codes used to simulate the COVID-19 model
in "A model for COVID-19 with isolation, quarantine, and testing as control
measures"
([https://doi.org/10.1016/j.epidem.2021.100437](https://doi.org/10.1016/j.epidem.2021.100437)),
written by M. S. Aronna, R. Guglielmi, and L. M. Moschen, and the codes used for the estimation of the underreporting rate of COVID-19 cases in
the city of Rio de Janeiro. 

## The Model 

### Abstract of the article

 In this article we propose a compartmental model for the dynamics of
 Coronavirus Disease 2019(COVID-19).  We take into account the presence of
 asymptomatic infections and the main policies that  have  been  adopted  so
 far  for  the  combat  of  this  disease:  isolation  (or  social
 distancing)  ofa portion of the population,  quarantine for confirmed cases
 and testing.  We model isolation by separating the population in two groups:
 one composed by key-workers that keep working during the  pandemic  and  have
 a  usual  contact  rate,  and  a  secondhttps://arxiv.org/abs/2005.07661  group  consisting  of  people  that
 are enforced/recommended to stay at home.  We refer to quarantine as strict
 isolation, and it is applied to confirmed infected cases.
 
 In the proposed model, the proportion of people in isolation, the level of
 contact reduction and the testing rate are control parameters that can vary in
 time, representing policies that evolve indifferent stages.  We obtain an
 explicit expression for the basic reproduction numberR0in terms of the
 parameters of the disease and of the control policies.  In this way we can
 quantify the effect that isolation and testing have in the evolution of the
 epidemic.  We present a series of simulations to illustrate different
 realistic scenarios.  From the expression ofR0and the simulations we
 conclude that isolation (social distancing) and testing among asymptomatic
 cases are fundamental actions to control the epidemic, and the stricter these
 measures are and the sooner they are implemented, the more lives can be saved.
 Additionally, we show that people that remain in isolation
 significantly reduce  their  probability  of  contagion,  so  risk  groups
 should  be  recommended  to  maintain  a  low contact rate during the course
 of the epidemic. 

### The graphic from model 

![Image from the model](images/model.svg)

### System of equations 

![Image from the system](images/equation.svg)

### Report of Scientific Initiation 

In the folder `notes`, one can find the report of my scientific initiation
(written in portuguese) with the following abstract:  

The COVID-19 disease caused by the SARS-CoV-2 virus has been spreading rapidly over the world since the beginning of 2020.  The understanding of its dynamics in the population is crucial to take measures that contain the spread. In this report, we consider the epidemiological model SEIAQR aforementioned to understand the beginning of the epidemic in the city of Rio de Janeiro and, in particular, the underreporting rate, that is, the proportion of infected individuals that the system didn't register. The curves of confirmed cases and deaths were adjusted to the actual city data using the error-weighted least squares method. We use B-splines to approximate the transmissibility and mortality of the disease. Also, we analyze the structural and practical identifiability of the model to verify the feasibility of the estimates. We used the Bootstrap method to quantify the uncertainty about the parameter's estimates. In the period March-July 2020, we obtain the point estimate of 0.9 for underreporting with a 95 \% confidence interval (0.85, 0.93). 

## Structure 

We organize it as follows: 

```{bash}
├── data
│   ├── covid_data_organized.csv
│   ├── parameters.yaml
├── experiments
├── identifiability
│   └── covid_model_id.txt
├── images
├── notebooks
│   ├── bootstrap_results.ipynb
│   ├── data_analysis.ipynb
│   └── model_fitting.ipynb
├── notes
│   ├── underreporting_estimation.pdf
├── pyscripts
│   ├── bootstrap.py
│   ├── dynamics_model.py
│   ├── estimation.py
│   ├── execute_model.py
│   ├── initial_estimation.py
│   └── scenarios.py
└── README.md
```

This structure includes:

* `data`: COVID-19 raw and organized data from Rio de Janeiro, parameters for simulations, and variables saved after the simulations; 
* `experiments`: bootstrap and fitting;
* `notebooks`: where we do the experiments. One should follow the order data analysis, model fitting, and bootstrap results;
* `pyscripts`: scripts written in Python. The explanation for each one is below. 
   1. `bootstrap.py`: parametric bootstrap used in the report;
   2. `dynamics_model.py`: differential equations from the model;
   3. `estimation.py`: estimates the overall confirmed case curve and death curve;
   4. `execute_model.py`: after changing the parameters in `parameter.yaml`, run this code to simulate the model;
   5. `initial_estimation.py`: estimates the initial curve of the confirmed cases;
   6. `scenarios.py`: calculate the scenarios explained in the article and save them into the `table_values.txt` file. 

## Hands on

We expect it is the most reproducible that it can be. To experiment, one needs: 

- Have Python 3 installed. One can check this
  [here](https://www.python.org/downloads/). 
- Clone this repository in your machine with the command ```git clone https://github.com/lucasmoschen/covid-19-model``` 
- Install the requirements with ```pip install -r requirements.txt```

## Experimenting 

After the steps above, so as to experiment the model, it's needed to follow
these steps: 

1. Change the parameters in the `parameters.yaml`. Maintain the format of the
   file as already is. The `change_p` parameter is reserved to the date when
   p change. It can be one, for now. 
2. Change the variable functions in the class `Parameters_Functions` in
   `dynamics_model.py` file. 
3. Enter in the `pyscripts` folder and run `python execute_model.py`. Follow
   the instructions to save the variables in the terminal. 

If you want to reproduce the results in the article or change the values and
see how it affects each scenario, it's necessary to follow these steps: 

1. Access the `scenarios.py` file and change the parameters values in your
   way. 
2. Enter in the `pyscripts` folder and run `python scenarios.py`. The result
   will be in `data/table_values.## Experimenting 

After the steps above, so as to experiment the model, it's needed to follow
these steps: 

1. Change the parameters in the `parameters.yaml`. Maintain the format of the
   file as already is. The `change_p` parameter is reserved to the date when
   $p$ change. It can be one, for now. 
2. Change the variable functions in the class `Parameters_Functions` in
   `dynamics_model.py` file. 
3. Enter in the `pyscripts` folder and run `python execute_model.py`. Follow
   the instructions to save the variables in the terminal. 

If you want to reproduce the results in the article or change the values and
see how it affects each scenario, it's necessary to follow these steps: 

1. Access the `scenarios.py` file and change the parameters values in your
   way. 
2. Enter in the `pyscripts` folder and run `python scenarios.py`. The result
   will be in `data/table_values.txt`. txt`. 

## Suggestions

Please, for any suggestions, write an issue, and I will answer as quickly as I can. Thanks!
