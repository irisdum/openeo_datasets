export python_version="3.10"
export name="mmdc_eo"
conda deactivate
if ! [ -z "$1" ]
then
    export name=$1
fi


export target=/home/dumeuri/virtualenv/$name
echo "$target"
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

conda activate
conda create --yes --prefix $target python==${python_version} pip
echo "create blakc env "
# Enter virtualenv
conda deactivate
conda activate $target
conda install --yes pytorch=2.0.0 torchvision -c pytorch
conda install -c conda-forge curl
which python
python --version



# Requirements
pip install -r environment.txt
python -m pip install "dask[distributed]" --upgrade
pip install 'python-lsp-server[all]'
pip install 'black[d]'
python -m ipykernel install --user --name "$name"
pip install tabulate

# End
conda deactivate
