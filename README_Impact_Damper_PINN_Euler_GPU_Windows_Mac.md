# Impact-Damper PINN on ETH Euler GPU

Terminal-first workflow for running the multi-impact PINN impact-damper chain on the ETH Euler cluster with GPU acceleration.

This guide is for users working from either:

- macOS / Linux terminal
- Windows PowerShell

The actual PINN training runs on ETH Euler, not locally.

Main files:

- `pinn_impact_chain_simulation.py`
- `pinn_impact_chain_solver.py`

---

# Project Structure

```text
PINN_Impact_Damper/
├── pinn_impact_chain_simulation.py
├── pinn_impact_chain_solver.py
├── Results_free_free_100s/
├── README_Impact_Damper_PINN_Euler_GPU.md
└── References/
```

---

# Part 1. First-time Setup on ETH Euler

## 1) Log in to Euler

### macOS / Linux terminal

```bash
ssh zharui@euler.ethz.ch
```

### Windows PowerShell

```powershell
ssh zharui@euler.ethz.ch
```

If login succeeds, you should see something like:

```bash
zharui@eu-login-xx:~$
```

---

## 2) Upload the project to Euler

Run this from your LOCAL computer, not inside Euler.

### macOS / Linux terminal

Example if the project is in `~/Documents/GitHub/`:

```bash
scp -r ~/Documents/GitHub/PINN_Impact_Damper/PINN zharui@euler.ethz.ch:~
```

### Windows PowerShell

Example if the project is in `C:\Users\Rui\Documents\GitHub\`:

```powershell
scp -r "C:\Users\zharui\Documents\GitHub\PINN_Impact_Damper\PINN" zharui@euler.ethz.ch:~
```

If your Windows username or folder is different, modify the path accordingly.

Then log in to Euler and check:

```bash
cd ~/PINN_Impact_Damper
ls
```

---

## 3) Load the Euler module stack

On Euler:

```bash
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
```

Check Python:

```bash
python --version
which python
```

Expected:

```text
Python 3.11.6
```

---

## 4) Request an interactive GPU node

```bash
srun \
  --time=08:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem-per-cpu=16G \
  --gpus=rtx_4090:1 \
  --pty bash
```

After allocation:

```bash
hostname
nvidia-smi
```

If `nvidia-smi` shows an NVIDIA GPU, the GPU node request worked.

---

## 5) Load modules again on the GPU node

After entering the GPU node:

```bash
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
```

---

## 6) Create a virtual environment

Only needed once:

```bash
python -m venv ~/venvs/pinn_impact_gpu
source ~/venvs/pinn_impact_gpu/bin/activate
```

---

## 7) Enable internet proxy for pip installation

On Euler compute nodes, use:

```bash
module load eth_proxy
```

---

## 8) Install required Python packages

```bash
pip install --upgrade pip
pip install numpy==1.26.4 scipy matplotlib h5py
pip install tensorflow
```

Optional:

```bash
pip install pandas jupyter
```

---

## 9) Check TensorFlow GPU visibility

```bash
python - <<'EOF'
import tensorflow as tf
print('TF version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('Visible GPUs:', tf.config.list_physical_devices('GPU'))
EOF
```

Expected:

- TensorFlow version is printed
- `Built with CUDA: True`
- at least one visible GPU

---

# Part 2. Run the Impact-Damper PINN on Euler

## 1) Go to the project directory

```bash
cd ~/PINN_Impact_Damper
```

---

## 2) Activate the environment

```bash
source ~/venvs/pinn_impact_gpu/bin/activate
```

---

## 3) Run the simulation

```bash
python PINN/pinn_impact_chain_simulation.py
```

This will:

1. Build the 20-DOF free-free chain
2. Run Newmark reference simulations
3. Train segment-by-segment PINNs
4. Detect impacts automatically
5. Propagate post-impact initial conditions
6. Compute dispersion maps
7. Save figures and results

---

# Part 3. Current Default Settings

The current simulation uses:

```python
n_dof = 20
T_segment = 1.0
n_t_segment = 100
nIter_per_segment = 1000
t_end_target = 100.0
```

Meaning:

- 20 structural DOFs
- 1-second PINN segment length
- 100 collocation points per segment
- 1000 Adam iterations per segment
- 100 seconds total simulation time

Velocity cases:

```python
input_velocity_cases = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}
```

---

# Part 4. Recommended Debug Settings

For fast debugging on Euler:

```python
n_dof = 5
T_segment = 0.5
n_t_segment = 50
nIter_per_segment = 200
t_end_target = 10.0
```

This is useful for:

- debugging impact detection
- validating PINN convergence
- checking energy conservation
- testing dispersion extraction

---

# Part 5. Recommended Publication Settings

For cleaner nonlinear dispersion studies:

```python
n_dof = 20-50
T_segment = 0.5-1.0
n_t_segment = 100-200
nIter_per_segment = 1000-5000
t_end_target = 20-50
```

Notes:

- 100 s is usually longer than necessary for dispersion extraction
- long simulations increase training cost substantially
- smaller segments often improve impact localization
- larger DOF counts improve k-space resolution in the FFT

---

# Part 6. Output Files on Euler

Results are written to:

```bash
Results_free_free_100s/
```

Generated outputs include:

```text
newmark_reference.png
pinn_diagnostics.png
dispersion_maps.png
batch_summary.csv
*.npz
*.mat
```

---

# Part 7. Download Results Back to Local Computer

Run this from your LOCAL computer, not inside Euler.

## macOS / Linux terminal

```bash
scp -r \
zharui@euler.ethz.ch:~/PINN_Impact_Damper/Results_free_free_100s \
~/Downloads/
```

Open locally on macOS:

```bash
open ~/Downloads/Results_free_free_100s
```

## Windows PowerShell

Example download to your Windows Downloads folder:

```powershell
scp -r zharui@euler.ethz.ch:~/PINN_Impact_Damper/Results_free_free_100s C:/Users/zharui/Downloads/
```

Then open the folder manually in File Explorer:

```powershell
explorer "$env:USERPROFILE\Downloads\Results_free_free_100s"
```

---

# Part 8. Upload Updated Files to Euler

Run this from your LOCAL computer.

## macOS / Linux terminal

Upload all Python files:

```bash
scp ~/Documents/GitHub/PINN_Impact_Damper/*.py \
zharui@euler.ethz.ch:~/PINN_Impact_Damper/
```

Upload one file only:

```bash
scp ~/Documents/GitHub/PINN_Impact_Damper/pinn_impact_chain_simulation.py \
zharui@euler.ethz.ch:~/PINN_Impact_Damper/
```

## Windows PowerShell

Upload all Python files:

```powershell
scp "C:\Users\Rui\Documents\GitHub\PINN_Impact_Damper\*.py" zharui@euler.ethz.ch:~/PINN_Impact_Damper/
```

Upload one file only:

```powershell
scp "C:\Users\Rui\Documents\GitHub\PINN_Impact_Damper\pinn_impact_chain_simulation.py" zharui@euler.ethz.ch:~/PINN_Impact_Damper/
```

---

# Part 9. Example SBATCH Job

On Euler, create a batch file:

```bash
nano run_impact_pinn.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=impact_pinn
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=impact_pinn.out

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source ~/venvs/pinn_impact_gpu/bin/activate

cd ~/PINN_Impact_Damper

python pinn_impact_chain_simulation.py
```

Submit:

```bash
sbatch run_impact_pinn.sh
```

Monitor:

```bash
squeue -u zharui
```

Check output:

```bash
cat impact_pinn.out
```

---

# Part 10. Next-Time Use

If the environment already exists, you only need:

```bash
ssh zharui@euler.ethz.ch
cd ~/PINN_Impact_Damper
srun --time=08:00:00 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G --gpus=1 --pty bash
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
source ~/venvs/pinn_impact_gpu/bin/activate
python pinn_impact_chain_simulation.py
```

---

# Notes

- Windows and macOS/Linux differ only in the local-side `scp` paths.
- All Euler-side commands are Linux commands and are the same for both systems.
- The workflow uses segment-by-segment PINN retraining.
- Each segment trains a new PINN using propagated initial conditions.
- Impact times are found using root-finding on the frozen PINN.
- Dispersion maps are computed using 2D FFT of the PINN response.
- Long simulations with many impacts can become computationally expensive.
- Euler GPU execution is recommended for 20+ DOF studies.
