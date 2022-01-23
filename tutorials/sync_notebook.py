"""
This script exists to solve a problem in jupyter book: If the path in the table
of contents has both a jupyter notebook and a markdown file, then jupyter book
will use the markdown file. This is quite annoying if you want to store outputs
manually in notebooks from long running jobs. So, the simple solution might be
to just only store a jupyter notebook. The downside to this approach is that
there are several useful things that can be done with MySTMarkdown and jupyter
book that cannot be done with just a notebook. The solution to this problem is
to store both a jupyter notebook and a markdown file, but store them in
different folders. This script helps to achieve that goal.

This script accepts an input that is either a jupyter notebook path or a
markdown path and syncs with its paired notebook. The markdown file will live in
a `md/` subfolder next to the jupyter notebook.

If a jupyter notebook path is passed, it syncs the file that is passed with
its corresponding markdown file in the md subfolder adjacent the notebook
and vice versa if a markdown path is passed.
"""
import sys
import os.path
import subprocess


input_filename = sys.argv[1]

dir, filename = os.path.split(input_filename)
filebase, ext = os.path.splitext(filename)

print(ext)
if ext == ".ipynb":
    md_dir = os.path.join(dir, "md")
    md_file = os.path.join(md_dir, filebase + ".md")

    cmd1 = ["jupytext", "-s", "-o", md_file, input_filename]
    print(cmd1)
    subprocess.run(cmd1)
elif ext == ".md":
    print("check this code in sync_notebook.py")
    print("check this code")
    print("check this code")
    print("check this code")
    print("check this code")
    import sys

    sys.exit()
    ipynb_dir = os.path.split(dir)[0]
    ipynb_file = os.path.join(ipynb_dir, filebase + ".ipynb")
    cmd2 = ["jupytext", "-s", "-o", ipynb_file, input_filename]
    print(cmd2)
    subprocess.run(cmd2)
