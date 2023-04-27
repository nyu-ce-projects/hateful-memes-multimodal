import numpy as np

single_gpu = open("hpml-1xRTX8000.out", "r").readlines()
multi_gpu = open("hpml-4xRTX8000.out", "r").readlines() 
single_gpu_v100 = open("hpml-1xV100.out", "r").readlines()

single_gpu = [line.split(",") for line in single_gpu if line.startswith("GPU : Q")]
single_gpu_core_util = [int(b.strip().split(":")[-1].strip().split("%")[0].strip()) for a,b,c,d in single_gpu]
single_gpu_mem_util = [int(c.strip().split(":")[-1].strip().split("%")[0].strip()) for a,b,c,d in single_gpu]

multi_gpu = [line.split(",") for line in multi_gpu if line.startswith("GPU : Q")]
multi_gpu_core_util = [int(b.strip().split(":")[-1].strip().split("%")[0].strip()) for a,b,c,d in multi_gpu]
multi_gpu_mem_util = [int(c.strip().split(":")[-1].strip().split("%")[0].strip()) for a,b,c,d in multi_gpu]

single_gpu_v100 = [line.split(",") for line in single_gpu_v100 if line.startswith("GPU : T")]
single_gpu_v100_core_util = [int(b.strip().split(":")[-1].strip().split("%")[0].strip()) for a,b,c,d in single_gpu_v100]
single_gpu_v100_mem_util = [int(c.strip().split(":")[-1].strip().split("%")[0].strip()) for a,b,c,d in single_gpu_v100]

print(np.mean(single_gpu_core_util), np.mean(single_gpu_mem_util))
print(np.mean(multi_gpu_core_util), np.mean(multi_gpu_mem_util))
print(np.mean(single_gpu_v100_core_util), np.mean(single_gpu_v100_mem_util))
