# Dependencies
pip install transformers
pip install pathlib
pip install statistics
pip install numpy

# Run command
sbatch torch/distributed/benchmarks/fsdp/launcher.slurm

# Output results
results are written into "delay/" folder

# Building and using PyTorch with NCCL and EFA
export WORK_DIR=${HOME}
cd $WORK_DIR/pytorch
USE_NCCL=1 python setup.py install
git clone -b aws https://github.com/aws/aws-ofi-nccl.git ${WORK_DIR}/aws-ofi-nccl-src
cd ${WORK_DIR}/aws-ofi-nccl-src
./autogen.sh
./configure --prefix=${WORK_DIR}/aws-ofi-nccl \
--with-mpi=/opt/amazon/openmpi \
--with-libfabric=/opt/amazon/efa \
--with-nccl=${WORK_DIR}/pytorch/build/nccl \
--with-cuda=$CUDA_HOME
make
make install

"if EFA is set up correctly, you should see logs like this: NCCL INFO NET/OFI Selected Provider is efa"
