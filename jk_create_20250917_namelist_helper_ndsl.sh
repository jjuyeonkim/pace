#!/usr/bin/env bash

set -e

module load miniforge gcc netcdf ncview openmpi-ucx hdf5 nccmp nco
module list

JJUYEONKIM_BRANCH=20250925_namelist_helper_ndsl # Namelist Refactor: ndsl namelist helper
# After talking with NASA/NOAA on 20250925, I'm looking to write generic helper functions
# to convert input.nml files into a dict that can eventually be used to create
# either a PhysicsConfig or DynamicalCoreConfig

read -p "Do you want to create a new conda environment today? [y/n]" yes_no

yes_no=${yes_no,,}
echo "You chose: ${yes_no}"

if [ "${yes_no}" == "y" ]; then   
    conda_env_name="jk_${JJUYEONKIM_BRANCH}"

    #echo "Removing conda env: " ${conda_env_name}
    #conda env remove --name ${conda_env_name} -y
    echo "Creating conda env: " ${conda_env_name}
    conda create -y -n ${conda_env_name} python=3.11
    echo "Environment created: ${conda_env_name}"
    echo Activating ${conda_env_name}
    conda activate ${conda_env_name}

    read -p "Enter to checkout repo: git clone git@github.com:jjuyeonkim/pace.git ..." && git clone git@github.com:jjuyeonkim/pace.git && cd pace

    read -p "Enter to update submodules... " && git submodule update --init --recursive
    read -p "Enter to update remote... " && git submodule update --remote

    read -p "Enter to delete original NDSL..." && rm -rf NDSL
    read -p "Enter to checkout repo: 'git clone git@github.com:jjuyeonkim/ndsl.git NDSL'..." && git clone git@github.com:jjuyeonkim/ndsl.git NDSL && cd NDSL
    read -p "Update NDSL submodules and install again..."
    git checkout ${JJUYEONKIM_BRANCH}
    git submodule update --init --recursive
    cd ..
    
    read -p "Enter to 'pip install -e .[dev]'... " && pip install -e .[dev]
    read -p "Enter to 'pip install -e .'... " && pip install -e .

    pip install -e NDSL

    read -p "Enter to 'pip install -e pyFV3'... " && pip install -e pyFV3
    read -p "Enter to 'pip install -e pySHiELD'... " && pip install -e pySHiELD
    read -p "Enter to 'pip install pytest'... " && pip install pytest
    
    read -p "Enter to 'pip install numpy==1.26.4'... " && pip install numpy==1.26.4

    read -p "Enter to create the input data..." && mkdir tests/main/input && python examples/generate_eta_files.py tests/main/input

    read -p "Enter to run pace unit tests..." && python -m pytest -v tests/main/
    
    read -p "Enter to run pySHiELD translate tests..." && cd pySHiELD
    python -m pytest tests/savepoint/ --data_path=~/pace2/pySHiELD/test_data/8.1.3/c12_6ranks_baroclinic/physics/ --backend=numpy --threshold_overrides_file=tests/savepoint/translate/overrides/standard.yaml
    cd ..

    read -p "Enter to run pyFV3 translate tests..." && cd pyFV3
    python -m pytest tests/savepoint/ --data_path=~/pace2/pyFV3/test_data/8.1.3/c12_6ranks_standard/dycore/ --backend=numpy --threshold_overrides_file=tests/savepoint/translate/overrides/standard.yaml
    cd ..
    
    #read -p "Enter to continue... " && conda deactivate
    #read -p "Enter to continue... " && conda remove --name ${conda_env_name} -y --all
else
    echo "New environment not created."
fi
