import sys
import os.path
import subprocess

ipynb = sys.argv[1]
dir, filename = os.path.split(ipynb)
filebase, ext = os.path.splitext(filename)
md_dir = os.path.join(dir, "md")
md_file = os.path.join(md_dir, filebase + ".md")

cmd1 = ["jupytext", "-s", "-o", md_file, ipynb]
print(cmd1)
subprocess.run(cmd1)

cmd2 = ["jupytext", "-s", "-o", ipynb, md_file]
print(cmd2)
subprocess.run(cmd2)
