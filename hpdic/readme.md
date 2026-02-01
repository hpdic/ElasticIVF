# HPDIC MOD of FAISS

# CloudLab Setup

## 5 nodes, each with 2 GPUs: `ibm8335` @ CloudLab Clem.

It's a bit tedious to setup 5 nodes with 10 GPUs each on CloudLab Clem. In fact, it's a bit challenging to install on IBM Power9 architecture. Among many other issues, here's a list of warnings I have for you: (i) You cannot use the default VS Code editor because Power9 is compatible with VS Code. (2) You will need to manually install a lot of dependencies (e.g., cmake 3.28+) since the default Ubuntu 20.04 apt repository is too old.

First we need to setup the cluster and smoke-test master node `node0`:
```bash
sudo chmod 777 /hpdic
ln -s /hpdic ~/hpdic
cd ~/hpdic
git clone git@github.com:hpdic/ElasticIVF.git
git config --global user.name "Dongfang Zhao"
git config --global user.email "dzhao@uw.edu"
# copy ssh key to ~/hpdic/ssh_cloudlab
chmod 600 ~/hpdic/ssh_cloudlab
cat <<EOF > ~/.ssh/config
Host *
    IdentityFile ~/hpdic/ssh_cloudlab
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
EOF
cat <<EOF > ~/hpdic/hosts
node0
node1
node2
node3
node4
EOF
for i in node0 node1 node2 node3 node4; do
    printf "Trying to reach %-8s ... " $i
    ssh $i "hostname -s"
done
sudo apt install pssh btop -y
parallel-ssh -h ~/hpdic/hosts -i "hostname"
grep -v "node0" ~/hpdic/hosts > ~/hpdic/workers
wget https://us.download.nvidia.com/tesla/470.161.03/NVIDIA-Linux-ppc64le-470.161.03.run
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nvidia-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nvidia-nouveau.conf
sudo update-initramfs -u
sudo reboot
# After reboot (wait for ~5 minutes), run the following command to install NVIDIA driver:
mkdir -p ~/hpdic/tmp
sudo bash ~/hpdic/NVIDIA-Linux-ppc64le-470.161.03.run --silent --dkms --no-x-check --no-nouveau-check --tmpdir=/hpdic/tmp
nvidia-smi # Then you'll see something like the following:
donzhao@node0:~$ nvidia-smi 
Thu Jan 29 23:00:40 2026       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-SXM2...  Off  | 00000002:01:00.0 Off |                    0 |
| N/A   27C    P0    41W / 300W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-SXM2...  Off  | 00000003:01:00.0 Off |                    0 |
| N/A   25C    P0    30W / 300W |      0MiB / 16280MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
donzhao@node0:~$ 
```

Then install NVIDIA drivers on worker nodes
```bash
parallel-ssh -h ~/hpdic/workers "echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nvidia-nouveau.conf && echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
parallel-ssh -h ~/hpdic/workers -t 0 "sudo update-initramfs -u"
parallel-ssh -h ~/hpdic/workers "sudo reboot"
# Wait for ~5 minutes for works to reboot
parallel-ssh -h ~/hpdic/workers "sudo mkdir -p /hpdic && sudo chmod 777 /hpdic && ln -s /hpdic ~/hpdic && mkdir -p ~/hpdic/tmp"
parallel-scp -h ~/hpdic/workers ~/hpdic/NVIDIA-Linux-ppc64le-470.161.03.run ~/hpdic/
parallel-ssh -h ~/hpdic/workers -t 0 "sudo sh /hpdic/NVIDIA-Linux-ppc64le-470.161.03.run --silent --dkms --no-x-check --no-nouveau-check --tmpdir=/hpdic/tmp"
parallel-ssh -h ~/hpdic/hosts -i "nvidia-smi -L" # Then you should see something like this on each worker:
donzhao@node0:~/hpdic$ parallel-ssh -h ~/hpdic/hosts -i "nvidia-smi -L" # Then you should see something like this on each worker:
[1] 23:30:41 [SUCCESS] node0
GPU 0: Tesla P100-SXM2-16GB (UUID: GPU-7344c36b-49e2-e2dd-ee41-747e5fb6378a)
GPU 1: Tesla P100-SXM2-16GB (UUID: GPU-7d830700-a263-c7a3-3562-f318060d56b8)
[2] 23:30:41 [SUCCESS] node1
GPU 0: Tesla P100-SXM2-16GB (UUID: GPU-3d7cc14a-4cfb-c00f-121a-45212c8a8764)
GPU 1: Tesla P100-SXM2-16GB (UUID: GPU-48cc4dcc-e87d-6f63-9053-8dfa96431426)
[3] 23:30:48 [SUCCESS] node3
GPU 0: Tesla P100-SXM2-16GB (UUID: GPU-4903be71-0134-478a-a43c-b9bb6072201f)
GPU 1: Tesla P100-SXM2-16GB (UUID: GPU-3006099e-4332-c5cc-37b4-b2ab9990b27d)
[4] 23:30:48 [SUCCESS] node2
GPU 0: Tesla P100-SXM2-16GB (UUID: GPU-5e134fe8-0bdb-16ff-b379-bf4059fd402e)
GPU 1: Tesla P100-SXM2-16GB (UUID: GPU-9803c684-90dc-e1d6-18e2-e1b91ef80b6d)
[5] 23:30:48 [SUCCESS] node4
GPU 0: Tesla P100-SXM2-16GB (UUID: GPU-7c660ff8-6cd6-7d07-eecd-d280f3a337a6)
GPU 1: Tesla P100-SXM2-16GB (UUID: GPU-2ff7cceb-c8df-c77c-79e6-58acbb1ef658)
donzhao@node0:~/hpdic$ 
```

Install tools (e.g., cuda, MPI) and SIVF on all nodes
```bash
cd ~/hpdic
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux_ppc64le.run
sudo mkdir -p /hpdic/usr_local
sudo cp -a /usr/local/* /hpdic/usr_local/
sudo mount --bind /hpdic/usr_local /usr/local
sudo bash ~/hpdic/cuda_11.4.4_470.82.01_linux_ppc64le.run --silent --toolkit --tmpdir=/hpdic/tmp
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev build-essential cmake git
which mpic++ && which nvcc
cd ~/hpdic/ElasticIVF
mkdir -p build && cd build
sudo apt install -y ccache swig libopenblas-dev
pip3 install numpy
sudo apt install -y libssl-dev
cd ~/hpdic
wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3.tar.gz
tar -zxvf cmake-3.28.3.tar.gz
cd cmake-3.28.3
./bootstrap --parallel=$(nproc) && make -j$(nproc)
sudo make install
hash -r
cmake --version
cd ~/hpdic/ElasticIVF
cmake -B build . \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="60"
# 60 (P100), 70 (V100), 75 (RTX6000), 80 (A100)
make -C build -j
cd ~/hpdic/ElasticIVF/build
make -j test_sivf_mpi_insert test_sivf_mpi_delete test_sivf_mpi_search
mpirun --allow-run-as-root -np 1 ./build/faiss/gpu/test_sivf_mpi_insert
mpirun --allow-run-as-root -np 2 ./build/faiss/gpu/test_sivf_mpi_insert

parallel-ssh -h ~/hpdic/workers -t 0 "sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev libopenblas-dev python3-numpy && mkdir -p ~/hpdic"
parallel-scp -h ~/hpdic/workers ~/hpdic/cuda_11.4.4_470.82.01_linux_ppc64le.run ~/hpdic/
parallel-ssh -h ~/hpdic/workers -t 0 "sudo bash /hpdic/cuda_11.4.4_470.82.01_linux_ppc64le.run --silent --toolkit --tmpdir=/hpdic/tmp"
parallel-scp -h ~/hpdic/workers -r ~/hpdic/ElasticIVF ~/hpdic/
mpirun --allow-run-as-root \
    -np 2 \
    --host node0,node1 \
    --map-by node \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_insert
mpirun --allow-run-as-root \
    -np 2 \
    --host node0:2 \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_insert    
mpirun --allow-run-as-root \
    -np 1 \
    --host node0 \
    --map-by node \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_insert    
mpirun --allow-run-as-root \
    -np 2 \
    --host node0,node1 \
    --map-by node \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_delete
mpirun --allow-run-as-root \
    -np 2 \
    --host node0:2 \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_delete 
mpirun --allow-run-as-root \
    -np 1 \
    --host node0 \
    --map-by node \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_delete 
mpirun --allow-run-as-root \
    -np 2 \
    --host node0,node1 \
    --map-by node \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_search
mpirun --allow-run-as-root \
    -np 2 \
    --host node0:2 \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_search      
mpirun --allow-run-as-root \
    -np 1 \
    --host node0 \
    --map-by node \
    --mca opal_cuda_support 0 \
    -x NCCL_P2P_DISABLE=1 \
    ./build/faiss/gpu/test_sivf_mpi_search     
```

## Single node with 4 GPUs: `c4130` @ CloudLab Wisc.

Setup GPUs
```bash
ln -s /hpdic ~/hpdic
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535 nvidia-cuda-toolkit
sudo apt install -y openmpi-bin libopenmpi-dev
sudo reboot
nvidia-smi
# Then you should see something like this:
donzhao@node0:~$ nvidia-smi 
Wed Jan 28 19:28:19 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 535.288.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla V100-SXM2-16GB           Off | 00000000:04:00.0 Off |                    0 |
| N/A   33C    P0              39W / 300W |      0MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2-16GB           Off | 00000000:06:00.0 Off |                    0 |
| N/A   31C    P0              40W / 300W |      0MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2-16GB           Off | 00000000:07:00.0 Off |                    0 |
| N/A   31C    P0              40W / 300W |      0MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2-16GB           Off | 00000000:08:00.0 Off |                    0 |
| N/A   32C    P0              39W / 300W |      0MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
donzhao@node0:~$ 
```

Install SIVF
```bash
sudo apt install python3.12-venv python3-dev -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0)}')"
cd ~/hpdic
git clone https://github.com/hpdic/ElasticIVF.git
cd ElasticIVF
python3 -m venv myenv
source myenv/bin/activate
python3 ~/hpdic/ElasticIVF/hpdic/script/test_gpu.py
cd ~/hpdic/ElasticIVF
cmake -B build . \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="70" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j
cd ~/ElasticIVF/tutorial/cpp
g++ 4-GPU.cpp -o 4-GPU.bin \
    -fopenmp \
    -I ../.. \
    -I /usr/local/cuda/include \
    -L ../../build/faiss -lfaiss \
    -L /usr/local/cuda/lib64 -lcudart -lcublas \
    -lopenblas \
    -Wl,-rpath=$(pwd)/../../build/faiss
./4-GPU.bin
```

Test SIVF on multiple GPUs:
```bash
# SIVF add benchmark:
cd ~/hpdic/ElasticIVF/build
make -j test_sivf_mpi_insert test_sivf_mpi_delete test_sivf_mpi_search
mpirun --allow-run-as-root -np 1 ./faiss/gpu/test_sivf_mpi_insert
mpirun --allow-run-as-root -np 2 ./faiss/gpu/test_sivf_mpi_insert
mpirun --allow-run-as-root -np 4 ./faiss/gpu/test_sivf_mpi_insert
mpirun --allow-run-as-root -np 1 ./faiss/gpu/test_sivf_mpi_delete
mpirun --allow-run-as-root -np 2 ./faiss/gpu/test_sivf_mpi_delete
mpirun --allow-run-as-root -np 4 ./faiss/gpu/test_sivf_mpi_delete
mpirun --allow-run-as-root -np 1 ./faiss/gpu/test_sivf_mpi_search
mpirun --allow-run-as-root -np 2 ./faiss/gpu/test_sivf_mpi_search
mpirun --allow-run-as-root -np 4 ./faiss/gpu/test_sivf_mpi_search
```

# Chameleon Cloud Setup

## Multi-Node GPU Cluster

After local compilation on the master node, sync the code to worker nodes and run MPI:
```bash
# Setup the network:
parallel-ssh -h hosts -i 'sudo systemctl stop firewalld && sudo systemctl disable firewalld'
parallel-ssh -h hosts -i 'sudo apt install -y cmake ccache swig libopenblas-dev libmkl-dev'
MY_CIDR=$(ip -o -4 addr show | grep "10.52" | awk '{print $4}')
echo "export OMPI_MCA_btl=tcp,self" >> ~/.bashrc
echo "export OMPI_MCA_btl_tcp_if_include=$MY_CIDR" >> ~/.bashrc
echo "export OMPI_MCA_oob_tcp_if_include=$MY_CIDR" >> ~/.bashrc
echo "export OMPI_MCA_opal_cuda_support=0" >> ~/.bashrc
source ~/.bashrc

# Sync code to worker nodes (faster than parallel-scp)
cd ~/hpdic/ElasticIVF
for host in $(cat ~/hpdic/workers); do
    rsync -avz --exclude 'data/' --exclude '.git/' --exclude '*.o' \
    ~/hpdic/ElasticIVF $host:~/hpdic/ &
done
wait
echo "Sync Complete!"

# MPI execution (synthetic data) across multiple GPU nodes:
cd ~/hpdic/ElasticIVF
mpirun --allow-run-as-root \
    -np 10 \
    --host gpu0:4,gpu1:4,gpu2:2 \
    -x LD_LIBRARY_PATH \
    ~/hpdic/ElasticIVF/build/faiss/gpu/test_sivf_mpi_insert
mpirun --allow-run-as-root \
    -np 10 \
    --host gpu0:4,gpu1:4,gpu2:2 \
    -x LD_LIBRARY_PATH \
    ~/hpdic/ElasticIVF/build/faiss/gpu/test_sivf_mpi_delete
mpirun --allow-run-as-root \
    -np 10 \
    --host gpu0:4,gpu1:4,gpu2:2 \
    -x LD_LIBRARY_PATH \
    ~/hpdic/ElasticIVF/build/faiss/gpu/test_sivf_mpi_search    

# MPI of DINO10B on 10 GPUs across 3 nodes:
bash ~/hpdic/ElasticIVF/hpdic/script/download_dino.sh
cd ~/hpdic/ElasticIVF/build
make -j test_sivf_dino_add
mpirun --allow-run-as-root \
    -np 10 \
    --host gpu0:4,gpu1:4,gpu2:2 \
    -x LD_LIBRARY_PATH \
    ~/hpdic/ElasticIVF/build/test_sivf_dino_add 
```

## Single-Node GPUs

We assume you are using (e.g., Chameleon Cloud `nc33` at U. Chicago) Ubuntu 24.04, NVIDIA RTX 6000 GPU (24 GB RAM, Driver 560.35.05, CUDA 12.6), 192 GB RAM, Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz (48 Cores).

### Recompile C++
```bash
# Recompile Faiss with HPDIC modifications:
cd ~/ElasticIVF
cmake -B build . \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j faiss

# SIVF add benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_add
./faiss/gpu/test_sivf_add

# SIVF search benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_search
./faiss/gpu/test_sivf_search

# SIVF delete benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_delete
./faiss/gpu/test_sivf_delete

# SIVF sensitivity benchmark:
cd ~/ElasticIVF/build
make -j test_sivf_sensitivity
./faiss/gpu/test_sivf_sensitivity

# Dataset Deep1B:
cd ~/ElasticIVF/build
make test_sivf_deep_add test_sivf_deep_search test_sivf_deep_delete test_sivf_deep_pareto -j
./test_sivf_deep_add
./test_sivf_deep_search
./test_sivf_deep_delete
./test_sivf_deep_pareto

# Dataset SIFT1M:
cd ~/ElasticIVF/build
make -j test_sivf_sift_add
./test_sivf_sift_add
make -j test_sivf_sift_search
./test_sivf_sift_search
make -j test_sivf_sift_delete
./test_sivf_sift_delete
make -j test_sivf_sift_pareto
./test_sivf_sift_pareto

# Dataset GIST1M:
cd ~/ElasticIVF/build
make test_sivf_gist_add test_sivf_gist_search test_sivf_gist_delete test_sivf_gist_pareto -j
./test_sivf_gist_add
./test_sivf_gist_delete
./test_sivf_gist_pareto
# ./test_sivf_gist_search
bash ~/ElasticIVF/hpdic/script/search_gist.sh

# Dataset T2I-1B:
cd ~/ElasticIVF/build
make test_sivf_t2i_add test_sivf_t2i_search test_sivf_t2i_delete test_sivf_t2i_pareto -j
./test_sivf_t2i_add
./test_sivf_t2i_delete
./test_sivf_t2i_pareto
# ./test_sivf_t2i_search
bash ~/ElasticIVF/hpdic/script/search_t2i.sh

# Sliding window:
cd ~/ElasticIVF/build
make -j test_sivf_sliding
./test_sivf_sliding # default is SIFT
./test_sivf_sliding gist

# Memory usage:
cd ~/ElasticIVF/build
make -j test_sivf_memory
./test_sivf_memory # default is SIFT
./test_sivf_memory gist

# SIVF vs. Non-IVF:
cd ~/ElasticIVF/build
make -j test_sivf_nonivf
./test_sivf_nonivf

# CAGRA:
source ~/ElasticIVF/myenv/bin/activate
pip install cuvs-cu12 --extra-index-url https://pypi.nvidia.com
pip install cupy-cuda12x
export LD_LIBRARY_PATH=$(dirname $(find ~/ElasticIVF/myenv -name "libcuvs_c.so" | head -n 1)):$LD_LIBRARY_PATH
cd ~/ElasticIVF/hpdic/script/
python bench_cagra.py

# Profiling
cd ~/ElasticIVF/build
make -j test_sivf_profiling
./test_sivf_profiling
sudo nsys profile     --trace=cuda,nvtx,osrt     --gpu-metrics-device=all     --output=sivf_pcie_evidence     --force-overwrite=true     ./test_sivf_profiling
sudo /usr/local/cuda/bin/ncu \
    --target-processes all \
    --metrics gpu__time_duration.sum,dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \ 
    --csv \
    --launch-skip 100 \
    --launch-count 10 \
    ./test_sivf_profiling > sivf_metrics_fast.csv
nsys stats sivf_pcie_evidence.nsys-rep --report gputrace,cuda_api_sum

# P99 latency test:
cd ~/ElasticIVF/build
make -j test_sivf_p99
./test_sivf_p99

# Mixed workload latency test:
cd ~/ElasticIVF/build
make -j test_sivf_mixed
./test_sivf_mixed

```

### Benchmarks
```bash
# Download SIFT1M dataset:
bash ~/ElasticIVF/hpdic/script/download_sift.sh

# Quick Python test with precompiled Faiss library:
source ~/ElasticIVF/myenv/bin/activate
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_rt.so python3 ~/ElasticIVF/hpdic/script/benchmark_baseline.py

# Serious C++ test with local compilation of Faiss source code:
cp ~/ElasticIVF/hpdic/config/c_cpp_properties.json ~/.vscode/.
cd ~/ElasticIVF/hpdic/experiment
g++ -O3 -std=c++17 -fopenmp benchmark_baseline.cpp -o benchmark_baseline.bin \
    -I/home/cc/ElasticIVF \
    -I/usr/local/cuda/include \
    -L/home/cc/ElasticIVF/build/faiss \
    -L/usr/local/cuda/lib64 \
    -lfaiss \
    -lopenblas \
    -lcudart \
    -lcublas
./benchmark_baseline.bin
```

### Installation
```bash
git config --global user.name "Dongfang Zhao"
git config --global user.email "dzhao@uw.edu"
sudo apt install -y cmake ccache swig g++ libopenblas-dev libmkl-dev git
cd ~
git clone https://github.com/hpdic/ElasticIVF.git
cd ElasticIVF
python3 -m venv myenv
source myenv/bin/activate
which python # e.g., /home/cc/ElasticIVF/myenv/bin/python
pip install matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 ~/ElasticIVF/hpdic/script/test_gpu.py
rm -rf build
cmake -B build . \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_RAFT=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DPython_EXECUTABLE=$(which python)
make -C build -j $(nproc)
cd build/faiss/python
python setup.py install
```

### Test
```bash
cd ~/ElasticIVF/tutorial/cpp
# For Intel MKL
g++ 4-GPU.cpp -o 4-GPU.bin \
    -fopenmp \
    -I ../.. \
    -I /usr/local/cuda/include \
    -L ../../build/faiss -lfaiss \
    -L /usr/local/cuda/lib64 -lcudart -lcublas \
    -lmkl_rt \
    -Wl,-rpath=$(pwd)/../../build/faiss
./4-GPU.bin

# For OpenBLAS
g++ 4-GPU.cpp -o 4-GPU.bin \
    -fopenmp \
    -I ../.. \
    -I /usr/local/cuda/include \
    -L ../../build/faiss -lfaiss \
    -L /usr/local/cuda/lib64 -lcudart -lcublas \
    -lopenblas \
    -Wl,-rpath=$(pwd)/../../build/faiss
./4-GPU.bin
```

### Reinstall Python package
```bash
cd ~/ElasticIVF/build/faiss/python
python setup.py install
```
