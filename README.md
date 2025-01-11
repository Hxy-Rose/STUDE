# STUDE
Improved  DCE-MRI parameter estimation via spatial-temporal information-driven unsupervised learning

# Getting Started
To create the environment dce in anaconda, the following command can be used:

> conda env create -f environment.yml

# Generate the numerical brain phantom
Synthesizing numerical brain data can be done using **simulate_data.py**.
> cd ./Numerical-brain-Code
> 
> python simulate_data.py

* Data is organized in a hierarchical folder structure as shown below.

```
./data
├─train.txt
├─test.txt
├─DATA_SD_7
      ├─train
          ├─SD_7_01.nii
          ├─SD_7_02.nii
          ├─SD_7_03.nii
          ├─SD_7_04.nii 
          └─SD_7_05.nii 
          ...
      ├─val
      └─test
├─DATA_SD_25
      ├─train
      ├─val
      └─test
        
└─DATA_SD_50
└─DATA_SD_100
...
```
