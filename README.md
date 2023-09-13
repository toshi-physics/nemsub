# nemsub_clean
code for nematic substrate alignment project

written for fenicsx

install fenicsx and its linked requirements from conda

running python 3.9

pde_hom_rho.py takes in parameters from file paramsnew.json, they both should be in the same directory or supply full file path to params json file in code.
pde_hom_rho.py saves dat files of Qxx, Qxy, and rho evaluated at gridpoints. There are also h5py files with mesh and full fields saved.
If you want to eval the fields at points other than mesh points, those changes need to me made in pde_hom_rho.py.
Data files saved in folders named with gamma, pi, and rhoseed values. All runs saved in the same folder with run number appended to file name.

vis_hom_rho.py takes in values of g, pi, rhoseed, and run number and visualises the time evolution of the fields. Python parameter prompter will help if you type --help.

ODE version is in a ipynb and doesn't require fenics obv.
