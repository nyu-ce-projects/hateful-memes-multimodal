import subprocess as sp
import os
from threading import Thread , Timer
import sched, time

def get_gpu_cores_mem_util():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory --format=csv"
    try:
        use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return use_info

def print_gpu_util_every_sec():
    """
        This function calls itself every 5 secs and print the gpu_memory.
    """
    Timer(1.0, print_gpu_util_every_sec).start()
    use_info = get_gpu_cores_mem_util()
    for item in use_info:
        name, core_util, mem_util = item.split(",")
        print("GPU : {}, GPU Core Util : {}, GPU Mem Util : {}, Time: {}".format(name, core_util, mem_util, time.perf_counter()))
