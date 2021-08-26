# UCLCHEM Surrogate

This code base requires that the Julia Language is Installed. Either 
version 1.6.1 or 1.6.2. Please install it through https://julialang.org/. Package manager version have been shown to leave binaries behind need for various packages.
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

## Structure
```
.
├── scripts                   # Examples of how to use ESNs, Surrogates and Chemical Networks along with scripts for plot creation
├── images                    # Images I included in the Paper
├── models                    # Binaries needed to run the code
├── src                       # Source files, containing the implementation of the gas phase network, ESNs, and surrogate model
├── output                    # output images from the code in scripts/ and Main.jl
├── Main.jl                   # Main Result producing script
└── README.md
```
## Instructions
1. Please unzip the binaries in the models/ folder. These contain the full network surrogate model. The simplified version was not included due to GitHub's 100MB file limit.
2. While in the root directory, simply run `julia Main.jl` to create the ESN and DESN Surrogate prediction for a random set of rates for the full network and to reproduce a reduced version of the heatmaps. If still in the REPL then run `using DrWatson; include(projectdir("Main.jl"))`
3. To create the box and whisker plots, please run `julia scripts/<problem_name>.jl` **Please note, this can take a very long time to completely run (hours). This is also using just a fraction of the samples of the plots in the paper so results will vary.**

## Notes
- All output is already located in the `output/` folder, so it is not required to run everything.
- The scripts directory provides minimalistic examples of how to use the chemical network, esn and surrogate should you need to. The other three scripts are dedicated entirely to producing the error plots from the paper.
- The heatmaps use a fraction of the samples and produce higher error than in the paper. After investigating this I determined it was due to the proximity of the rates to the interpolation samples. They are not the same, but are goemetrically close. I've made note of this in the paper.
- In `Main.jl` the heatmap portion has a parameter to change the number of samples of temperature or density. If the number of samples exceeds ~25, the program will crash. This is a bug in some part of the solver. I have confirmed that the problem that causes the exception to be thrown is not unsolvable; a fresh Julia session will allow the problem to be solved, but this means we cannot accumulate samples without writing to disk and restarting Julia repeatedly.  
