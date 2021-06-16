# BIE Book 

Using JupyterBook!

### Process

1. `mamba env create`
1. Install PyCUDA manually `mamba install -y pycuda`.
1. Run the `./dev` script to rebuild whenever sources change. 
1. Open [localhost:8000](localhost:8000) to view the built book. I recommend the "Live Reload" Firefox extension to auto-reload the page when changes are made.
1. Write code using Jupyter Notebook or Lab. Do fancier MyST Markdown editing using the raw markdown text file.
1. Add to `references.bib`.
1. Run notebooks through black via the traditional Jupyter Notebook interface. 
1. Run `./publish` to copy built files into my website and freeze the conda environment.
