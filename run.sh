# check if there is a conda environment named optimized-gemm
# env="optimized-gemm"
# if ! conda env list | grep -q "$env"; then
#     conda env create -f environment.yml -n $env
# fi
# conda activate $env

# export MKL_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# lscpu | grep "Model name:"
# lscpu | grep -A 3 "L1d cache:"

# export PREFIX=$(pwd);
# rm -rf $PREFIX/build/; mkdir $PREFIX/build/; cd $PREFIX/build/;
# echo "CONDA_PREFIX: $CONDA_PREFIX"

# cmake ..; make VERBOSE=1 -j4; cd -

# ff=("avx2-8x4-unroll.x" "avx2-cache-blocking-8x4-unroll.x")
# ff=("naive-jki.x" "naive-ijk.x" "naive-kji.x")
ff=("dgemm-96-1024-192.x" "dgemm-96-1024-384.x" "dgemm-96-1024-768.x" "dgemm-192-1024-192.x" "dgemm-192-1024-384.x" "dgemm-192-1024-768.x" "dgemm-384-1024-192.x" "dgemm-384-1024-384.x" "dgemm-384-1024-768.x")

cd $PREFIX/build/; echo "" > $PREFIX/plot/tmp;
for i in $(seq 1 40); do
    echo ""
    echo "L = $i * 64"
    for f in "${ff[@]}"; do
        l=$(($i * 64))

        echo "Running $f with arguments $l ..."
        echo "Running $f with arguments $l ..." >> $PREFIX/plot/tmp
        ./$f "$l" "4" >> $PREFIX/plot/tmp
        echo "" >> $PREFIX/plot/tmp
    done
done

cd -

python $PREFIX/plot/collect.py $PREFIX/plot/tmp $PREFIX/plot/out.log
python $PREFIX/plot/plot.py $PREFIX/plot/out.log $PREFIX/plot/out.png
