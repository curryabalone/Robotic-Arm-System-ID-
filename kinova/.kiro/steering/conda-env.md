---
inclusion: fileMatch
fileMatchPattern: "kinova/**"
---

# Conda Environment

All Python code in the `kinova/` folder must be run under the conda environment `7dof`.

For scripts with MuJoCo viewer on macOS, use mjpython from the conda env:
```bash
conda run -n 7dof mjpython script.py
```

For non-viewer scripts:
```bash
conda run -n 7dof python script.py
```
