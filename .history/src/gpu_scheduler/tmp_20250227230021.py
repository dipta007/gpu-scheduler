import threading
import queue
import time
from typing import List
import logging

class GPUJobScheduler:
    def __init__(self, available_gpus: List[int]):
        self.available_gpus = queue.Queue()
        for gpu in available_gpus:
            self.available_gpus.put(gpu)
        self.total_gpus = len(available_gpus)
        self.job_queue = queue.Queue()
        self.active_jobs = {}
        self.lock = threading.Lock()
        
        self.results = []

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        
    def add_job(self, job_id, func, num_gpus=1, **kwargs):
        """Add a job to the queue with specified GPU requirement."""
        if num_gpus > self.total_gpus:
            error_msg = f"Job {job_id} requires {num_gpus} GPUs, but only {self.total_gpus} are available in total"
            logging.error(error_msg)
            self.results.append((job_id, f"Error: {error_msg}"))
            return False
            
        self.job_queue.put((job_id, func, num_gpus, kwargs))
        logging.info(f"Added job {job_id} to queue (requires {num_gpus} GPUs)")
        return True
        
    def run_job(self, job_id, func, gpu_ids, **kwargs):
        """Run a single job on the specified GPUs."""
        try:
            logging.info(f"Starting job {job_id} on GPUs {gpu_ids}")
            res = func(**kwargs, gpu_ids=gpu_ids)
            logging.info(f"Job {job_id} completed successfully on GPUs {gpu_ids}")
        except Exception as e:
            logging.error(f"Error running job {job_id}: {str(e)}")
            res = f"Error: {str(e)}"
            
        finally:
            # Return GPUs to available pool
            with self.lock:
                for gpu_id in gpu_ids:
                    self.available_gpus.put(gpu_id)
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                
                self.results.append((job_id, res))
            logging.info(f"GPUs {gpu_ids} are now available")
            
    def start_scheduler(self):
        """Start the scheduler to process jobs."""
        logging.info("Starting GPU job scheduler")
        
        while True:
            try:
                # Get next job from queue
                job_id, func, num_gpus, kwargs = self.job_queue.get_nowait()
                
                # Check if we have enough GPUs available
                if self.available_gpus.qsize() < num_gpus:
                    # Not enough GPUs available right now, put the job back in the queue
                    self.job_queue.put((job_id, func, num_gpus, kwargs))
                    time.sleep(0.5)  # Wait a bit before trying again
                    continue
                
                # Acquire the required number of GPUs
                gpu_ids = []
                for _ in range(num_gpus):
                    gpu_ids.append(self.available_gpus.get())

                logging.debug(f"job_id: {job_id}, func: {func}, gpu_ids: {gpu_ids}, kwargs: {kwargs}")
                
                # Start job in new thread
                with self.lock:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
                    thread = threading.Thread(target=self.run_job, args=(job_id, func, gpu_ids), kwargs=kwargs)
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


def func(model_id, x, gpu_ids):
    print("model_id", model_id)
    print("x", x)
    print(f"Using GPUs: {gpu_ids}")
    import time
    time.sleep(10)  # Simulate job running
    return model_id, x, gpu_ids

# Example usage
if __name__ == "__main__":
    # Initialize scheduler with available GPUs
    scheduler = GPUJobScheduler([0, 1, 2, 3])  # 4 GPUs numbered 0-3
    
    # Add example model training jobs with different GPU requirements
    scheduler.add_job(0, func, num_gpus=1, model_id="model_small", x=0)
    scheduler.add_job(1, func, num_gpus=2, model_id="model_medium", x=1)
    scheduler.add_job(2, func, num_gpus=4, model_id="model_large", x=2)
    scheduler.add_job(3, func, num_gpus=1, model_id="model_small_2", x=3)
    scheduler.add_job(4, func, num_gpus=2, model_id="model_medium_2", x=4)
    scheduler.add_job(5, func, num_gpus=3, model_id="model_large_2", x=5)
    
    # This job will fail because it requires more GPUs than are available
    scheduler.add_job(999, func, num_gpus=5, model_id="model_too_big", x=999)
    
    # Start processing jobs
    scheduler.start_scheduler()

    print("Job results:")
    for job_id, result in scheduler.results:
        print(f"Job {job_id}: {result}")