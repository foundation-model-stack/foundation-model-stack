# Run a Simple Triton Program with Debugger Support

To run through this code with the Triton Debugger, please use the following command:

~~~
TRITON_INTERPRET=1 python vector_add.py    
~~~

# Run a Simple Triton Program with NSight Compute Profiler

First, download the Nvidia Nsight Compute System to your local system here: https://developer.nvidia.com/nsight-compute

If you are running this tool in a remote machine, you must make sure that tool is installed there as well. This can be achieved through conda:

~~~
conda install -c nvidia nsight-compute
~~~

Once both systems are setup, to profile the kernel:

~~~
ncu --target-processes all 
--set detailed 
--import-source yes 
--section SchedulerStats 
--section WarpStateStats 
--section SpeedOfLight_RooflineChart 
--section SpeedOfLight_HierarchicalTensorRooflineChart 
-o output_file_location 
python vector_add.py
~~~

Download the trace file that ncu generates to your local machine to see full detailed analysis.