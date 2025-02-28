import subprocess
import threading
import queue
import time
from typing import List, Callable
import logging

class GPUJobScheduler:
    def __init__(self, available_gpus: List[int]):
        self.available_gpus = queue.Queue()
        for gpu in available_gpus:
            self.available_gpus.put(gpu)
        self.job_queue = queue.Queue()
        self.active_jobs = {}
        self.lock = threading.Lock()
        
        self.results = []

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        
    def add_job(self, job_id, func, **kwargs):
        """Add a job to the queue."""
        self.job_queue.put((job_id, func, kwargs))
        logging.info(f"Added job {job_id} to queue")
        
    def run_job(self, job_id, func, gpu_id, **kwargs):
        """Run a single job on the specified GPU."""
        try:
            logging.info(f"Starting job {job_id} on GPU {gpu_id}")
            res = func(**kwargs, gpu_id=gpu_id)
            logging.info(f"Job {job_id} completed successfully on GPU {gpu_id}")
        except Exception as e:
            logging.error(f"Error running job {job_id}: {str(e)}")
            res = f"Error: {str(e)}"
            
        finally:
            # Return GPU to available pool
            with self.lock:
                self.available_gpus.put(gpu_id)
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                
                self.results.append((job_id, res))
            logging.info(f"GPU {gpu_id} is now available")
            
    def start_scheduler(self):
        """Start the scheduler to process jobs."""
        logging.info("Starting GPU job scheduler")
        
        while True:
            try:
                # Get next job from queue
                job_id, func, kwargs = self.job_queue.get_nowait()
                
                # Wait for available GPU
                gpu_id = self.available_gpus.get()

                logging.debug(f"job_id: {job_id}, func: {func}, gpu_id: {gpu_id}, kwargs: {kwargs}")
                
                # Start job in new thread
                with self.lock:
                    thread = threading.Thread(target=self.run_job, args=(job_id, func, gpu_id), kwargs=kwargs)
                    self.active_jobs[job_id] = thread
                    thread.start()
                    
            except queue.Empty:
                # No more jobs in queue
                if not self.active_jobs:
                    break
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Scheduler error: {str(e)}")
                
        logging.info("All jobs completed")


def func(model_id, x, gpu_id):
    print("model_id", model_id)
    print("x", x)
    import time
    time.sleep(10)
    return model_id, x

# Example usage
if __name__ == "__main__":
    # Initialize scheduler with available GPUs
    scheduler = GPUJobScheduler([0, 1, 2, 3])  # 4 GPUs numbered 0-3
    
    # Add 10 example model training jobs
    for i in range(10):
        scheduler.add_job(i, func, model_id=f"model_{i}", x=i)
    
    # Start processing jobs
    scheduler.start_scheduler()

    print(scheduler.results)