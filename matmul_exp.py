import torch
import time 
import numpy as np

cpu_N=1<<9
cuda_N=1<<13

def test_np_matmul():
    print('A@B')
    c=0
    for _ in range(10):
        A=np.random.normal(size=(cpu_N,cpu_N))
        B=np.random.normal(size=(cpu_N,cpu_N))
        t0=time.time()
        np.matmul(A,B)
        c+=time.time()-t0
    print(f'{c:.5f}s')

    print('A@B.T')
    c=0
    for _ in range(10):
        A=np.random.normal(size=(cpu_N,cpu_N))
        B=np.random.normal(size=(cpu_N,cpu_N))
        t0=time.time()
        np.matmul(A,B.T)
        c+=time.time()-t0
    print(f'{c:.5f}s')

    print('A.T@B')
    for _ in range(10):
        A=np.random.normal(size=(cpu_N,cpu_N))
        B=np.random.normal(size=(cpu_N,cpu_N))
        t0=time.time()
        np.matmul(A.T,B)
        c+=time.time()-t0
    print(f'{c:.5f}s')

    print('A.T@B.T')
    for _ in range(10):
        A=np.random.normal(size=(cpu_N,cpu_N))
        B=np.random.normal(size=(cpu_N,cpu_N))
        t0=time.time()
        np.matmul(A.T,B.T)
        c+=time.time()-t0
    print(f'{c:.5f}s')


def test_torch_matmul():
    print('A@B')
    c=0
    for _ in range(10):
        A=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        B=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        t0=time.time()
        torch.matmul(A,B)
        c+=time.time()-t0
    print(f'{c:.5f}s')

    print('A@B.T')
    c=0
    for _ in range(10):
        A=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        B=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        t0=time.time()
        torch.matmul(A,B.T)
        c+=time.time()-t0
    print(f'{c:.5f}s')

    print('A.T@B')
    c=0
    for _ in range(10):
        A=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        B=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        t0=time.time()
        torch.matmul(A.T,B)
        c+=time.time()-t0
    print(f'{c:.5f}s')

    print('A.T@B.T')
    c=0
    for _ in range(10):
        A=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        B=torch.normal(0, 1, size=(cuda_N,cuda_N), device='cuda')
        t0=time.time()
        torch.matmul(A.T,B.T)
        c+=time.time()-t0
    print(f'{c:.5f}s')

if __name__=='__main__':
    test_np_matmul()
    print('CUDA')
    test_torch_matmul()
