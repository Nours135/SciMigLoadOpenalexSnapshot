import subprocess
import time
import torch


def clear_gpu_memory():
    torch.cuda.empty_cache()


def get_gpu_memory_allocated(gpu_id):
    if isinstance(gpu_id, torch.device):
        gpu_id = gpu_id.index
    elif isinstance(gpu_id, str):   
        if gpu_id.startswith('cuda:'):
            gpu_id = int(gpu_id.split(':')[1])
        elif gpu_id.isdigit():
            gpu_id = int(gpu_id)
        else:   
            raise ValueError(f"Invalid GPU ID format: {gpu_id}")

    try:
        shell_bash = f'nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits'
        output = subprocess.check_output(shell_bash, shell=True, encoding='utf-8')
        usage = int(output.strip().split('\n')[0])
        return usage
    except Exception as e:
        print(f"Failed to get GPU usage for GPU {gpu_id}: {e}")
        return None

def get_gpu_memory_usage(gpu_id=0):
    """Returns the GPU memory usage percentage using nvidia-smi."""
    if isinstance(gpu_id, torch.device):
        gpu_id = gpu_id.index
    elif isinstance(gpu_id, str):   
        if gpu_id.startswith('cuda:'):
            gpu_id = int(gpu_id.split(':')[1])
        elif gpu_id.isdigit():
            gpu_id = int(gpu_id)
        else:   
            raise ValueError(f"Invalid GPU ID format: {gpu_id}")
        
    try:
        # Run nvidia-smi to get GPU memory usage information
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Error getting GPU usage: {result.stderr}")
        
        # Parse memory usage from the output
        used_memory, total_memory = map(int, result.stdout.split(','))
        memory_usage_percentage = (used_memory / total_memory) * 100
        return memory_usage_percentage
    
    except Exception as e:
        print(f"Error occurred while checking GPU memory usage: {e}")
        return None

def wait_until_gpu_free(gpu_id=0, threshold=80, check_interval=1):
    """
    Waits until GPU memory usage drops below a certain threshold.
    
    Args:
        gpu_id (int): The GPU to monitor (default is 0).
        threshold (float): The memory usage percentage threshold (default is 80%).
        check_interval (int): How frequently to check memory usage (in seconds).
    """
    while True:
        memory_usage = get_gpu_memory_usage(gpu_id)
        if memory_usage is None:
            print("Failed to get GPU memory usage.")
            break
        
        print(f"GPU {gpu_id} memory usage: {memory_usage:.2f}%")
        
        if memory_usage < threshold:
            print(f"GPU memory usage is below {threshold}%, continuing.")
            break
        
        print(f"Waiting for GPU memory usage to drop below {threshold}%.")
        time.sleep(check_interval)


def select_device(device: str = 'gpu'):
    # 获取每张 GPU 的显存占用情况
    if device == 'gpu':
        for device_id in [0, 1]:
            if torch.cuda.is_available():
                allocated_memory = get_gpu_memory_allocated(device_id) * 1024**2
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                # import pdb; pdb.set_trace()
                free_memory = total_memory - allocated_memory
                # import pdb; pdb.set_trace()
                # 判断是否有足够的空闲显存
                if free_memory > 20 * 1024**3:  # 假设2GB内存足够跑模型
                    print(f"Using GPU {device_id} with {free_memory / 1024**3:.2f} GB free memory")
                    return torch.device(f"cuda:{device_id}")
    
    # 如果没有足够的空闲 GPU，使用 CPU
    print("Using CPU")
    return torch.device("cpu")


# Example usage: Wait until GPU 0's memory usage is below 80%
if __name__ == "__main__":
    res = select_device(device='gpu')
    print(res)
    wait_until_gpu_free(gpu_id=res, threshold=30, check_interval=20)
