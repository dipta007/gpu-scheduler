# GPU Scheduler

A simple tool to schedule gpu resources.

Let's say you have a server with 4 GPUs and you want to run 10 jobs (each job requires 1 GPU). They can finish any time and you don't want to sit in front of the server to run one after another, You can use this tool to schedule the jobs and it will run the jobs as soon as needed gpu is available.

## Installation

```bash
pip install gpu-scheduler
```

## Usage

```python
import time
from gpu_scheduler import GPUScheduler

def hf_with_given_gpu_ids(model_id, gpu_ids: list):
    device_map = {str(i): f"cuda:{gpu_id}" for i, gpu_id in enumerate(gpu_ids)}
    model = AutoModel.from_pretrained(model_id, device_map=device_map)
    # ...
    return model

def func(model_id, gpu_ids: list):
    """
    The function must accept gpu_ids as an argument
    This is the list of GPU IDs that the job will run on
    You have to manually set the GPU IDs in your code, i.e., above `hf_with_given_gpu_ids` function
    """
    time.sleep(4)  # Simulate job running
    return model_id, gpu_ids

if __name__ == "__main__":
    # Initialize scheduler with available GPUs
    scheduler = GPUScheduler([0, 1, 2, 3])  # 4 GPUs numbered 0-3

    # Add example model training jobs with different GPU requirements
    scheduler.add_job(0, func, num_gpus=1, model_id="model_small")
    scheduler.add_job(1, func, num_gpus=2, model_id="model_medium")
    scheduler.add_job(2, func, num_gpus=4, model_id="model_large")
    scheduler.add_job(3, func, num_gpus=1, model_id="model_small_2")
    scheduler.add_job(4, func, num_gpus=2, model_id="model_medium_2")
    scheduler.add_job(5, func, num_gpus=3, model_id="model_large_2")

    # This job will fail because it requires more GPUs than are available
    scheduler.add_job(999, func, num_gpus=5, model_id="model_too_big")

    # Start processing jobs
    scheduler.start_scheduler()

    print("Job results:")
    for job_id, result in scheduler.results:
        print(f"{job_id=}: {result=}")
```

### Output
```
Job results:
job_id=999: result='Error: Job 999 requires 5 GPUs, but only 4 are available in total'
job_id=1: result=('model_medium', [1, 2])
job_id=0: result=('model_small', [0])
job_id=3: result=('model_small_2', [3])
job_id=5: result=('model_large_2', [1, 2, 0])
job_id=2: result=('model_large', [3, 1, 2, 0])
job_id=4: result=('model_medium_2', [3, 1])
```