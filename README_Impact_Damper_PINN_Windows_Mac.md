# Impact-Damper Chain PINN — Local Setup on Windows and macOS

Terminal-first workflow for running the local impact-damper PINN simulation on Windows or macOS.

This guide is for the files:

```text
Impact_Damper_PINN/
├── pinn_impact_chain_simulation.py
├── pinn_impact_chain_solver.py
└── README_Impact_Damper_PINN_Windows_Mac.md
```

The main script runs an end-to-end 20-DOF free-free impact-damper chain simulation:

1. Build mass, damping, and stiffness matrices
2. Generate Newmark-beta reference trajectories
3. Run segment-by-segment multi-impact PINN simulation
4. Detect and propagate impact events
5. Plot diagnostics
6. Compute dispersion maps
7. Save NPZ, MAT, CSV, and PNG results

Default output folder:

```text
Results_free_free_100s/
```

---

## Current default simulation settings

In `pinn_impact_chain_simulation.py`, the default settings are:

```python
n_dof = 20
m_x   = 1.0
m_y   = 0.3
k     = 1.0
c     = 0.0
D     = 1.0
r     = 1.0
```

Input velocity cases:

```python
input_velocity_cases = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}
```

PINN training settings:

```python
T_segment          = 1.0
n_t_segment        = 100
nIter_per_segment  = 1000
t_end_target       = 100.0
optimizer_LB_value = True
```

Newmark reference settings:

```python
T_ref  = 100.0
dt_ref = 0.001
```

For debugging, reduce the problem first:

```python
n_dof = 5
t_end_target = 10.0
T_ref = 10.0
nIter_per_segment = 200
optimizer_LB_value = False
```

Then return to the full case after confirming the code works.

---

# Part 1. Windows setup

## 1) Install Miniconda or Anaconda

Recommended: install **Miniconda** for Windows.

After installation, open:

```text
Anaconda Prompt
```

or:

```text
Windows PowerShell
```

If using PowerShell, make sure `conda` is available.

---

## 2) Go to the project folder

Example:

```powershell
cd C:\Users\YourName\Documents\GitHub\Impact_Damper_PINN
```

Check that the files are there:

```powershell
dir
```

You should see:

```text
pinn_impact_chain_simulation.py
pinn_impact_chain_solver.py
```

---

## 3) Create a conda environment

```powershell
conda create -n impact_pinn python=3.10 -y
conda activate impact_pinn
```

Python 3.10 is recommended for TensorFlow compatibility.

---

## 4) Install packages

For normal CPU use:

```powershell
python -m pip install --upgrade pip
pip install numpy scipy matplotlib tensorflow
```

`scipy` is needed for root finding, L-BFGS-B, Newmark-related utilities, and optional MATLAB `.mat` export.

---

## 5) Check TensorFlow

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

For most Windows local runs, CPU TensorFlow is enough for debugging.

---

## 6) Run the simulation

```powershell
python pinn_impact_chain_simulation.py
```

The script will print progress for each case and each segment, for example:

```text
Running case: low
seg 001: impact #01 ...
```

---

## 7) Check results

```powershell
dir Results_free_free_100s
```

Expected files include:

```text
newmark_reference.png
pinn_diagnostics.png
dispersion_maps.png
batch_summary.csv
pinn_free_free_20dof_low.npz
pinn_free_free_20dof_medium.npz
pinn_free_free_20dof_high.npz
```

If `scipy.io.savemat` is available, `.mat` files will also be saved.

---

## 8) Open result folder

```powershell
explorer Results_free_free_100s
```

---

# Part 2. macOS setup

There are two common macOS cases:

1. Apple Silicon Mac: M1, M2, M3, M4
2. Intel Mac

For stability, start with CPU mode. After that, you can try Apple Metal acceleration.

---

## 1) Install Miniconda or Miniforge

For Apple Silicon, **Miniforge** is usually recommended.

After installation, open Terminal.

---

## 2) Go to the project folder

Example:

```bash
cd ~/Documents/GitHub/Impact_Damper_PINN
```

Check files:

```bash
ls
```

You should see:

```text
pinn_impact_chain_simulation.py
pinn_impact_chain_solver.py
```

---

## 3) Create a conda environment

```bash
conda create -n impact_pinn python=3.10 -y
conda activate impact_pinn
```

---

## 4) Install packages: simple CPU version

This is the safest first installation:

```bash
python -m pip install --upgrade pip
pip install numpy scipy matplotlib tensorflow
```

Check TensorFlow:

```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('Devices:', tf.config.list_physical_devices())"
```

---

## 5) Optional: Apple Silicon GPU / Metal acceleration

For M1/M2/M3/M4 Macs, you can try:

```bash
pip install tensorflow-macos tensorflow-metal
```

Then check GPU visibility:

```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

If this creates compatibility problems, remove the environment and use the CPU installation instead.

---

## 6) Run the simulation

```bash
python pinn_impact_chain_simulation.py
```

---

## 7) Check results

```bash
ls Results_free_free_100s
```

Open the result folder:

```bash
open Results_free_free_100s
```

---

# Part 3. Recommended quick smoke test

The full default case can be slow because it uses:

```python
n_dof = 20
t_end_target = 100.0
nIter_per_segment = 1000
optimizer_LB_value = True
```

For a first local test, edit `pinn_impact_chain_simulation.py`:

```python
n_dof = 5
layers = [1, 64, n_dof]
T_segment = 1.0
n_t_segment = 50
nIter_per_segment = 200
t_end_target = 10.0
T_ref = 10.0
optimizer_LB_value = False
save_dir = 'Results_free_free_smoke_test'
```

Then run:

```bash
python pinn_impact_chain_simulation.py
```

On Windows PowerShell, use the same command:

```powershell
python pinn_impact_chain_simulation.py
```

If this works, increase the problem gradually:

```python
n_dof = 10
nIter_per_segment = 500
t_end_target = 20.0
```

Then finally return to:

```python
n_dof = 20
t_end_target = 100.0
nIter_per_segment = 1000
optimizer_LB_value = True
```

---

# Part 4. What the code does

## Main simulation script

```text
pinn_impact_chain_simulation.py
```

This script controls the full workflow:

- defines system parameters
- defines velocity cases
- builds free-free chain matrices
- runs Newmark reference
- runs PINN segment-by-segment
- detects impacts
- updates post-impact velocities
- saves figures and data

Run it with:

```bash
python pinn_impact_chain_simulation.py
```

---

## Solver file

```text
pinn_impact_chain_solver.py
```

This file contains the main solver tools:

- `PIPNNs`
- `build_free_free_chain_matrices`
- `make_left_velocity_ic`
- `find_impact_times`
- `propagate_ics`
- `newmark_beta`

Usually, you do not run this file directly.

---

# Part 5. Output files

After a successful run, the result folder contains:

## Figures

```text
newmark_reference.png
pinn_diagnostics.png
dispersion_maps.png
```

## Data

```text
batch_summary.csv
pinn_free_free_20dof_low.npz
pinn_free_free_20dof_medium.npz
pinn_free_free_20dof_high.npz
```

Optional MATLAB files:

```text
pinn_free_free_20dof_low.mat
pinn_free_free_20dof_medium.mat
pinn_free_free_20dof_high.mat
```

---

# Part 6. Common problems

## Problem 1: `ModuleNotFoundError: No module named 'tensorflow'`

Activate the environment and install TensorFlow:

```bash
conda activate impact_pinn
pip install tensorflow
```

Windows PowerShell:

```powershell
conda activate impact_pinn
pip install tensorflow
```

---

## Problem 2: `ModuleNotFoundError: No module named 'pinn_impact_chain_solver'`

Make sure both files are in the same folder:

```text
pinn_impact_chain_simulation.py
pinn_impact_chain_solver.py
```

Then run the script from that folder.

---

## Problem 3: The run is too slow

Use the smoke-test settings first:

```python
n_dof = 5
t_end_target = 10.0
nIter_per_segment = 200
optimizer_LB_value = False
```

The L-BFGS step can be expensive, especially for long multi-segment simulations.

---

## Problem 4: The result folder is not created

Check whether the script stopped early due to an error.

Also check the value of:

```python
save_dir = 'Results_free_free_100s'
```

---

## Problem 5: Apple Silicon TensorFlow issues

Use the CPU installation first:

```bash
conda create -n impact_pinn python=3.10 -y
conda activate impact_pinn
pip install numpy scipy matplotlib tensorflow
```

Only try `tensorflow-macos` and `tensorflow-metal` after the CPU version works.

---

# Part 7. Notes for research use

For clean validation, do not directly rely on the full nonlinear dispersion plot first.

Recommended order:

1. Verify the no-impact Newmark reference
2. Run a small 5-DOF PINN case
3. Check displacement, velocity, and energy histories
4. Check detected impact times
5. Compare PINN response against a reference impact solver if available
6. Then compute dispersion maps
7. Finally test amplitude dependence using low, medium, and high velocity cases

The current default case is useful for generating nonlinear impact-dispersion results, but it should be validated carefully before being used as publication evidence.

---

# Part 8. Useful commands summary

## Windows

```powershell
cd C:\Users\YourName\Documents\GitHub\Impact_Damper_PINN
conda create -n impact_pinn python=3.10 -y
conda activate impact_pinn
pip install numpy scipy matplotlib tensorflow
python pinn_impact_chain_simulation.py
explorer Results_free_free_100s
```

## macOS

```bash
cd ~/Documents/GitHub/Impact_Damper_PINN
conda create -n impact_pinn python=3.10 -y
conda activate impact_pinn
pip install numpy scipy matplotlib tensorflow
python pinn_impact_chain_simulation.py
open Results_free_free_100s
```
