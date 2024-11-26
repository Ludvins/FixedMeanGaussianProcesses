# Fixed Mean Gaussian Processes

Recently, there has been an increasing interest in performing *post-hoc* uncertainty estimation about the predictions of pre-trained deep neural networks (DNNs).  Given a pre-trained DNN via back-propagation, post-hoc uncertainty estimation methods simply  provide a confidence of the DNN in the predictions made. This process enhances the predictions of the DNN with associated error bars, without deteriorating the prediction accuracy. With such a goal, we introduce here a new family of sparse variational Gaussian processes (GPs), where the posterior mean is fixed to any continuous function, when a universal kernel is used. We intentionally fix such a mean to the output of the DNN. Thus, our approach allows for effectively learning the predictive variances of a GP with the same predictive mean as the output of the DNN. The resulting GP predictive variances estimate the DNN prediction uncertainty. Our approach allows for efficient optimization using variational inference (VI), enabling stochastic optimization, with training costs that remain independent of the number of training points, scaling to very large datasets such as ImageNet. The proposed method, fixed mean GP (FMGP), is architecture-agnostic, relying solely on the outputs of the pre-trained model to adjust the predictive variances. Experimental results demonstrate improved performance in both uncertainty estimation and computational efficiency with respect to state-of-the-art methods.

MAP  |  LLA | FMGP
:-------------------------:|:-------------------------:|:-:|
![](demos\plots\synthetic_regression_map.png)  |  ![](demos\plots\synthetic_regression_lla.png) | ![](demos\plots\synthetic_regression_fmgp.png)
MFVI  | GP | HMC
![](demos\plots\synthetic_regression_mfvi.png)  |  ![](demos\plots\synthetic_regression_gp.png) | ![](demos\plots\synthetic_regression_hmc.png)

## Requirements

The used version of Python was [![Python](https://img.shields.io/badge/Python_3.11.9-blue)](https://www.python.org/downloads/release/python-3119/) with 
 [![Pytorch](https://img.shields.io/badge/PyTorch_2.2-purple)](https://pytorch.org/get-started/previous-versions/).

To create an environment and reproduce the experiments perform the following actions:
1. Create environment as `python -m venv .venv`
2. Activate environment as `source .venv/bin/activate`
3. Update pip as `pip install --upgrade pip`
4. Install requirements as `pip install -r requirements.txt`


## Folder Structure

The repository is structured in folders as:

    .
    ├── data                    # Contains used datasets (default download folder)
    ├── demos                   # Jupyter Notebooks for synthetic experiments
    │   ├── plots               # Stores the plots form the notebooks
    │   ├── synthetic.ipynb     # Notebook for Figure 1 
    ├── results                 # Obtained results in .csv format
    │   └─ *.csv
    ├── scripts                 # Python scripts for experiments
    │   ├── regression      
    │   │   └─ *.py  
    │   ├── cifar10
    │   │   └─ *.py  
    │   ├── imagenet
    │   │   └─ *.py  
    │   └── qm9
    │   │   └─ *.py  
    ├── bayesipy                # Code for different methods
    │   ├─ fmgp
    │   ├─ laplace
    │   ├─ mfvi
    │   ├─ sngp
    │   └─ utils
    ├── LICENSE
    ├── README.md
    └── requirements.txt

> [!Important]  
> Only Synthetic and Airline datasets are in the `data` folder by default as they can be difficult to obtain. The rest automatically download in the folder when needed.


## Usage

For a pre-trained map solution `f`, create an instance of FMGP as
```python
fmgp = FMGP(
    model=copy.deepcopy(f),  # Copy of MAP-trained model
    likelihood="regression",  # Regression setting
    kernel="RBF",  # RBF kernel for GP
    inducing_locations="kmeans",  # Use k-means to select inducing points
    num_inducing=10,  # Number of inducing points
    noise_variance=np.exp(-5),  # Initial noise variance
    subrogate_regularizer=True,  # Use subrogate regularizer
    y_mean=0,  # Mean of target
    y_std=1,  # Standard deviation of target
)
```
Train FMGP model with specified learning rate and iterations
```python
loss = fmgp.fit(iterations=70000, lr=0.001, train_loader=train_loader, verbose=True)
```

Predictions can be easily made as
```python
f_mean, f_var = fmgp.predict(torch.tensor(inputs))
```

## Experiments reproducibility


### Regression



```python
python ./scripts/regression/[method].py --dataset Year --seed 0
```

> [!Important]  
> Only `Year`, `Airline` and `Taxi` datasets can be used in these scripts. The available methods are: `lla`, `ella`, `valla`, `map` and `fmgp`. 


### Cifar10


```python
python ./scripts/cifar10/[method].py --seed 0
```

> [!Important]  
> The available methods are: `lla`, `ella`, `valla`, `map`, `mfvi`, `sngp` and `fmgp`. 

### Imagenet


```python
python ./scripts/imagenet/[method].py --seed 0
```

> [!Important]  
> The available methods are: `ella`, `map`, `mfvi` and `fmgp`. 

### QM9


```python
python ./scripts/qm9/[method].py --seed 0
```

> [!Important]  
> The available methods are: `lla`, `ella`, `map` and `fmgp`. The map solution can be obtained using `train_map.py`.
