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

    cmd1 = ["jupytext", "-s", "-o", md_file, ipynb]
    print(cmd1)
    subprocess.run(cmd1)
elif ext == ".md":
    cmd2 = ["jupytext", "-s", "-o", ipynb, md_file]
    print(cmd2)
    subprocess.run(cmd2)
