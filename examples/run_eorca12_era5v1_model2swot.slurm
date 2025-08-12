#!/bin/bash
#SBATCH --job-name=swot_eorca12_npd_
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10GB

#SBATCH --account=my_account
#SBATCH --partition=serial
#SBATCH --qos=serial

# ==============================================================
# run_npd_eorca12_era5v1_swot.slurm
#
# Description: SLURM script to perform SWOT OMIP interpolation
# using NOC eORCA12 ERA-5v1 2024 Near-Present Day outputs.
#
# Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
# Created On: 2025-08-11
#
# ==============================================================
# -- Input arguments to model2SWOT -- #
# Define path to eORCA12 monthly mean output directory:
filedir_mdl=/path/to/my/eORCA12/model/output/2024/
# Define filepath to eORCA12 domain_cfg file:
filepath_domain=/path/to/my/domain_cfg.nc

# Define path to SWOT global grid directory:
filedir_swot=/path/to/my/global_swot_grid_2024
# Define output directory:
filedir_out=/path/to/my/npd_eorca12_era5v1_global_swot_grid_2024

# Define chosen interpolation method:
interpolator="pyinterp_interpolator"

# Define core variable names:
lon_name="glamt"
lat_name="gphit"
time_name="time_counter"
ssh_name="zos"

# -- Python Environment -- #
# Activate Python virtual environment:
source /path/to/my/miniforge3/bin/activate
conda activate env_swot

# -- Interpolate eORCA12 1-day mean outputs onto SWOT grid -- #
# Iterating over cycles:
for cycle in {008..026}
do
    # Create output directory for the cycle if it doesn't exist:
    if [ ! -d  ${filedir_out}/cycle_${cycle} ]; then
        mkdir -p "${filedir_out}/cycle_${cycle}"
    fi

    # Define path to SWOT cycle directory:
    filedir_swot_cycle=${filedir_swot}/cycle_${cycle}/
    for file in $(ls ${filedir_swot_cycle})
    do
        echo "Interpolating eORCA12 ERA-5 v1 SSH onto SWOT grid ${file}"

        # Determine YYYYMM date-string of SWOT grid input:
        date=$(echo ${file} | grep -Eo '[[:digit:]]{4}[[:digit:]]{2}' | head -1)
        # Define path to nearest eORCA12 T1d monthly model file:
        fpath_mdl=$(echo ${filedir_mdl}/eORCA12_1d_grid_T_${date}-${date}.nc)
        # Define path to interpolated output file:
        fpath_out="${filedir_out}/cycle_${cycle}/${file/SWOT_GRID_L3_LR/eORCA12_ERA5v1_SWOT}"
        # Run model2swot interpolator:
        model2swot -m ${fpath_mdl} -k ${filepath_domain} -s ${filedir_swot_cycle}${file} -o ${fpath_out} -i ${interpolator} --model_lat_var ${lat_name} --model_lon_var ${lon_name} --model_time_var ${time_name} --model_ssh_var ${ssh_name}

        echo "Completed: Interpolated eORCA12 ERA-5 v1 SSH onto SWOT grid."
    done
done
