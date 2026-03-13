perf record -F 99 -e cycles:u --call-graph dwarf,16384 -p <PID> -o <ourtputfile.data> -- sleep 120
perf script --symfs ~/profile/cpu-vllm-sandbox/ --no-inline -i <ourtputfile.data> \
 | ~/profile/FlameGraph/stackcollapse-perf.pl --all \
 | ~/profile/FlameGraph/flamegraph.pl --hash > <perf_outputfile.svg>
# memory profiling
# page fault states 
perf stat -e page-faults -I 1000 -p <PID>
# memory record
perf record -e page-faults:u --call-graph dwarf,16384 -p 3154198 -o /home/bing/profile/flamegraph_results/perf_results/full_attention_in_16_out_256_bsz_1_pagefaults.data -- sleep 120
# memory script
perf script -i /home/bing/profile/flamegraph_results/perf_results/full_attention_in_16_out_256_bsz_1_pagefaults.data | ~/profile/FlameGraph/stackcollapse-perf.pl --all | ~/profile/FlameGraph/flamegraph.pl --color=mem --title="PyTorch Page Fault Flame Graph" --countname="pages" > /home/bing/profile/flamegraph_results/svg_results/full_attention_in_16_out_256_bsz_1_pagefaults.svg