for i in {1..100}
do
    ./test_bench_o/gemv_oneDNN_bf16 >> ./test_bench_o/results_oneDNN.log
done