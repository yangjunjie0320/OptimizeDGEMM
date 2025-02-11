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

opt_flags="-O3 -msse3 -mfpmath=sse -fomit-frame-pointer"
cmake -DCMAKE_CXX_FLAGS="$opt_flags" -DCMAKE_C_FLAGS="$opt_flags" ..
make VERBOSE=1 -j16; cd -

# ff=("naive-nkm.x") #  "macro-kernel-nkm.x" "macro-kernel-4x4.x" "micro-kernel-4x4.x")
ff=("naive-nkm.x" "pack-test.x" "pack-1.x")

cd $PREFIX/build/; echo "" > $PREFIX/plot/tmp; pwd
for i in $(seq 1 20); do
    echo ""
    export l=$(($i * 32))
    echo "L = $l"

    for f in ${ff[@]}; do
        echo "Running $f with arguments $l ..."
        # echo "Running $f with arguments $l ..." >> $PREFIX/plot/tmp
        # tail -n 1 $PREFIX/plot/tmp
        ./$f "$l" "4" # >> $PREFIX/plot/tmp
        # tail -n 1 $PREFIX/plot/tmp
        # echo "" >> $PREFIX/plot/tmp
    done
done

cd -

python $PREFIX/plot/collect.py $PREFIX/plot/tmp $PREFIX/plot/out.log
python $PREFIX/plot/plot.py $PREFIX/plot/out.log $PREFIX/plot/out.png

