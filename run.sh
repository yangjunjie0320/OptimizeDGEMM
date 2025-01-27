# check if there is a conda environment named optimized-gemm
env="optimized-gemm"
if ! conda env list | grep -q "$env"; then
    conda env create -f environment.yml -n $env
fi
conda activate $env

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

lscpu | grep "Model name:"
lscpu | grep -A 3 "L1d cache:"

export PREFIX=$(pwd);
rm -rf $PREFIX/build/; mkdir $PREFIX/build/; cd $PREFIX/build/;
echo "CONDA_PREFIX: $CONDA_PREFIX"

cmake ..; make VERBOSE=1 -j4; cd -

ff=("main-dgemm-blas")

cd $PREFIX/build/; echo "" > $PREFIX/plot/tmp;
for i in $(seq 1 10); do
    for f in "${ff[@]}"; do
        l=$(($i * 128))

        echo "Running $f with arguments $l ..."
        echo "Running $f with arguments $l ..." >> $PREFIX/plot/tmp
        ./$f "$l" "10" >> $PREFIX/plot/tmp
        echo "" >> $PREFIX/plot/tmp
    done
done

cd -

python $PREFIX/plot/collect.py $PREFIX/plot/tmp $PREFIX/plot/out.log
python $PREFIX/plot/plot.py $PREFIX/plot/out.log $PREFIX/plot/out.png
