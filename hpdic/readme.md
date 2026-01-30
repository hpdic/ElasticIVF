# HPDIC MOD of FAISS

## CloudLab Setup

### 5 nodes, each with 2 GPUs: `ibm8335` @ CloudLab Clem.
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
for i in node1 node2 node3 node4; do
    printf "Trying to reach %-8s ... " $i
    ssh $i "hostname -s"
done
```

### Single node with 4 GPUs: `c4130` @ CloudLab Wisc.

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


## Chameleon Cloud Setup

We assume you are using (e.g., Chameleon Cloud `nc33` at U. Chicago) Ubuntu 24.04, NVIDIA RTX 6000 GPU (24 GB RAM, Driver 560.35.05, CUDA 12.6), 192 GB RAM, Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz (48 Cores).

## Recompile C++
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
./test_sivf_gist_search
./test_sivf_gist_delete
./test_sivf_gist_pareto

# Dataset T2I-1B:
cd ~/ElasticIVF/build
make test_sivf_t2i_add test_sivf_t2i_search test_sivf_t2i_delete test_sivf_t2i_pareto -j
./test_sivf_t2i_add
./test_sivf_t2i_search
./test_sivf_t2i_delete
./test_sivf_t2i_pareto

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

## Benchmarks
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

## Installation
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

## Test
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

## Reinstall Python package
```bash
cd ~/ElasticIVF/build/faiss/python
python setup.py install
```
