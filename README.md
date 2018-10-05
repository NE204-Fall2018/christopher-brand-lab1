# NE 204 Report Lab1

# Digital Signal Processing for Gamma-Ray Spectroscopy in HPGe

Data was collected with a Coaxial HPGe detector with  sources.

A python script was written to perform an energy calibration for the data
sources.

Instructions for running the lab report are posted below:

## File instructions

### Downloading the data

Use the makefile to download the data for the lab:
Note: The make data command is currently commented out because of the large size of the data files. 
	If you wish to download them open the Makefile and uncomment the 'data' section

```
make data

```
### Generate Optimized Paramenters
Use the makefile to run the code to find optimzed parameters for the rise time, k, and gap time, m, for the trapizoidal filter 
as well as for finding the decay constant, tau, for use as M for the filter.

Note: The parameters k and m take about 20mins to run for a given data set of ~80,000 signals
	The code for finding tau is not optimized for speed and so takes ~40mins to run across ~80,000 signals. 

Note2: If you wish to run these indepedently the python scripts are found in: code/find_k.py|find_m.py|find_Tau.py 

```
make parameters
```

### Generate Graphs

Use the makefile to generate the signal graphs and calibrated spectrum graphs for the report.

```
make analysis
```


### Generating the final report in pdf format

```
make
```

Notes: [1] scripts generated using  python 3.5
