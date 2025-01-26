# check if there is a conda environment named optimized-gemm
env="optimized-gemm"
if ! conda env list | grep -q "$env"; then
    conda env create -f environment.yml -n $env
fi
conda activate $env

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

lscpu | grep "Model name:"
lscpu | grep -A 3 "L1d cache:"

export PREFIX=$(pwd);
rm -rf $PREFIX/build/; mkdir $PREFIX/build/; cd $PREFIX/build/;
echo "CONDA_PREFIX: $CONDA_PREFIX"

cmake ..; make VERBOSE=1 -j4; cd -

ff=("main-sgemm-naive-kji" "main-sgemm-block-32" "main-sgemm-block-64" "main-sgemm-block-128")

cd $PREFIX/build/;
for i in $(seq 1 40); do
    for f in "${ff[@]}"; do
        l=$(($i * 64))

        echo ""
        echo "Running $f with arguments $l ..."
        ./$f "$l" "6"
        echo ""
    done
done

cd -
