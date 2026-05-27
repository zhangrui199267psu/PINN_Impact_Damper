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
cd ~/PINN_Impact_Damper
```

---

### 3) Prepare the code

```bash
vim pinn_impact.sbatch
```

```bash
#!/bin/bash
#SBATCH --job-name=impact_pinn
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/impact_pinn_%x_%j.out
#SBATCH --error=logs/impact_pinn_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH -A es_chatzi

set -euo pipefail

# Usage:
#   sbatch run_impact_pinn_case.sh low
#   sbatch run_impact_pinn_case.sh medium
#   sbatch run_impact_pinn_case.sh high
#   sbatch run_impact_pinn_case.sh all
# Default: low
CASE=${1:-high}

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source ~/venvs/pinn_impact_gpu/bin/activate

cd ~/PINN_CFD_FSI_unsteady

cd ~/PINN_Impact_Damper

echo "Running impact PINN case: ${CASE}"
echo "Start time: $(date)"

python -u PINN/pinn_impact_chain_simulation_cli_revised.py --case "${CASE}"

echo "End time: $(date)"
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
sbatch pinn_impact.sbatch
```
See jobs:
```bash
squeue
```

See outputs
```bash
tail -f logs/impact_pinn_impact_pinn_912451.out
```
```bash
tail -f logs/impact_pinn_impact_pinn_912451.err
```

Cancel
```bash
scancel 912451
```
### 5) Check output files on Euler

After the run finishes, check:

```bash
ls Results_free_free_100s
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
