

export python_version="3.10"
export name="mmdc_eo"
if ! [ -z "$1" ]
then
    export name=$1
fi

source ~/set_proxy.sh
if [ -z "$https_proxy" ]
then
    echo "Please set https_proxy environment variable before running this script"
    exit 1
fi

export target=/work/scratch/$USER/virtualenv/$name

if ! [ -z "$2" ]
then
    export target="$2/$name"
fi

echo "Installing $name in $target ..."

if [ -d "$target" ]; then
   echo "Cleaning previous conda env"
   rm -rf $target
fi

# Create blank virtualenv
module purge
module load conda
conda activate
conda create --yes --prefix $target python==${python_version} pip

# Enter virtualenv
conda deactivate
conda activate $target

which python
python --version

conda deactivate
conda activate $target

# Requirements
pip install -r environment.txt
module unload git
python -m pip install "dask[distributed]" --upgrade
pip install 'python-lsp-server[all]'
python -m ipykernel install --user --name $export_name
pip install tabulate

# End
conda deactivate
