# Analysis
> This directory contains all scripts used for data analysis.

## Description
This directory contains all scripts used for data analysis.

## Requirements

### Dependencies
Install the Python dependencies defined in the requirements.txt.
```bash
pip install -r requirements.txt
```

#### Java parser
Set up the java-universal-parser by following the instructions in the README.md file in the java-universal-parser directory.

### Kernel setup
#### Jupyter in the browser
To create a new IPython kernel:
```bash
python -m ipykernel install
```

##### Jupyter in VSCode
Install the Python extension and select the kernel using `ctrl+shift+p`. Python will automatically create a new kernel for you. You can verify this with `jupyter kernelspec list`

### LaTeX
Matplotlib's `pgf` support requires a recent [LaTeX](http://www.tug.org/) installation that includes the TikZ/PGF packages (such as [TeXLive](http://www.tug.org/texlive/))

### Docker with gVisor
To run the analysis in a gVisor container, you need to have Docker installed. You can install Docker by following the instructions on the [Docker website](https://docs.docker.com/get-docker/). 
