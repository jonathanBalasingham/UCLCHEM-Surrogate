# UCLCHEM Surrogate

This code base requires that the Julia Language is Installed. Either 
version 1.6.1 or 1.6.2. Please install it through 
> UCLCHEM Surrogate

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box. 
```
.
├── scripts                   # Examples of how to use ESNs, Surrogates and Chemical Networks along with scripts for plot creation
├── images                    # Images I included in the Paper
├── models                    # Binaries needed to run the code
├── src                       # Source files, containing the implementation of the gas phase network, ESNs, and surrogate model
├── output                    # output images from the code in scripts/
├── Main.jl                   # Main Result producing script
└── README.md
```
## Instructions
1. Please unzip the binaries in the models/ folder. These contain the full network surrogate model. The simplified version was not included due to GitHub's 100MB file limit.
2. 


