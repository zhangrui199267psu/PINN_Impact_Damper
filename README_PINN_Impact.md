# PINN Impact-Damper on ETH Euler

### 1) Upload the project to Euler

If the project exists only on your local computer, upload it from your **local computer terminal**:

```bash
scp -r ~/Documents/GitHub/PINN_Impact_Damper zharui@euler.ethz.ch:~
```
Upload folder
```bash
scp -r ~/Documents/GitHub/PINN_Impact_Damper/PINN zharui@euler.ethz.ch:~/PINN_Impact_Damper/
```
Upload specific files (not folder)
```bash
scp ~/Documents/GitHub/PINN_Impact_Damper/PINN/*.py zharui@euler.ethz.ch:~/PINN_Impact_Damper/PINN/
```
Upload everything EXCEPT subfolders
```bash
scp ~/Documents/GitHub/PINN_Impact_Damper/PINN/* zharui@euler.ethz.ch:~/PINN_Impact_Damper/PINN/
```
### 2) Log in to Euler

Run this on your **local computer terminal**:

```bash
ssh zharui@euler.ethz.ch
```

When prompted for password, enter:

```text
XXXXXX
```

If login is successful, you should see something like:

```bash
zharui@eu-login-xx:~$
```

Then log in again if needed, and check on Euler:

```bash
ls
```

Delete file or floder:

```bash
rm x.py
```

```bash
rm -r x
```

```bash
cd ~/PINN_CFD_FSI_unsteady
```

---

### 3) Prepare the code

```bash
vim pinn_fsi1_steady.sbatch
```

```bash
#!/bin/bash
#SBATCH --job-name=pinn_fsi1
#SBATCH --output=logs/pinn_fsi1_%j.out
#SBATCH --error=logs/pinn_fsi1_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH -A es_chatzi

cd ~/PINN_CFD_FSI_unsteady

mkdir -p logs

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source ~/venvs/pinn_tf_gpu/bin/activate

export PYTHONUNBUFFERED=1

python -u PINN_FSI_Steady/unified_steady_fsi_final.py \
  --fe-data FE/fsi1_fsi_data.npz \
  --rho-f 1000 \
  --mu-f 1.0 \
  --um 0.2 \
  --rho-s 1000 \
  --nu-s 0.4 \
  --mu-s 5e5 \
  --adam-iters 30000 \
  --adam-lr 5e-4 \
  --lbfgs-maxiter 50000 \
  --lbfgs-maxfun 100000 \
  --lbfgs-maxcor 50 \
  --lbfgs-maxls 50 \
  --hidden-layers 4 \
  --hidden-width 64 \
  --n-fluid 80000 \
  --n-solid 20000 \
  --n-interface 12000 \
  --n-fix 3000\
  --w-iface-vel 20 \
  --w-iface-trac 0.05 \
  --w-solid-vel 1 \
  --w-detJ 0.1\
  --w-data-d 50 \
  --w-data-v 50 \
  --w-data-p 0.001 \
  --output-dir Results_PINN_FSI_SteadyFSI/FSI1
```

Edit:
```bash
i
```

Exit: (Esc - :wq - Enter)
```bash
:wq
```

### 4) Run the code
Submit:
```bash
sbatch pinn_fsi1_steady.sbatch
```
See jobs:
```bash
squeue
```

See outputs
```bash
tail -f logs/pinn_fsi1_912753.out
```
```bash
tail -f logs/pinn_fsi1_912753.err
```

Cancel
```bash
scancel 912753
```
### 5) Check output files on Euler

After the run finishes, check:

```bash
ls Results_PINN_FSI_SteadyFSI/FSI1
```

---

### 6) Download results back to your local computer


Exit to **local computer terminal**:

```bash
exit
```

Run on your **local computer terminal**:

```bash
scp -r zharui@euler.ethz.ch:~/PINN_CFD_FSI_unsteady/Results_PINN_FSI_SteadyFSI/FSI1\
  ~/Downloads
```

---
