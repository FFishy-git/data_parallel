import os
import time
import logging
import multiprocessing as mp
from queue import Empty
from typing import List, Any, Iterator, Generator, Type, Callable
from vllm.utils import get_open_port
import torch
from dataclasses import dataclass, field
# from vllm import LLM, SamplingParams

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# --- Structured Configuration using Dataclasses ---

@dataclass
class WorkerConfig:
    """Base configuration for a worker."""
    pass

@dataclass
class PipelineConfig:
    """Configuration for the DataParallelPipeline."""
    # basic gpu config
    num_gpus: int = field(default_factory=torch.cuda.device_count)
    gpus_per_worker: int = 1
    
    # queue config
    max_queue_size: int = 256  # Default value, can be overridden when instantiating
    fetching_task_timeout: float = 1800.0
    fetching_result_timeout: float = 600.0
    worker_setup_timeout: float = 1800.0
    sequential_worker_setup: bool = False   # if True, the workers will be setup sequentially, otherwise, they will be setup in parallel

# --- Universal Data Iterator Class ---

class DataIterator:
    """
    A factory class to create configured, batched iterators from various dataset types.

    This class encapsulates the logic for handling different data sources and
    applying preprocessing, making it easy to configure and reuse data iteration logic.
    Instances of this class are directly iterable.
    """
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        preprocess_fn: Callable[[Any], Any] = None, 
        first_n_batches: int | None = None
    ):
        """
        Initializes the DataIterator.

        Args:
            dataset (Any): The dataset to process. Supported types are:
                           - list
                           - torch.utils.data.Dataset
                           - datasets.Dataset (from Hugging Face)
                           - tf.data.Dataset (from TensorFlow)
            batch_size (int): The size of batches to yield.
            preprocess_fn (Callable, optional): A function to apply to each item
                                                 before batching. Defaults to None.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else lambda x: x
        self.first_n_batches = first_n_batches
        
        
    def __iter__(self) -> Iterator[List[Any]]:
        """
        Creates and returns the batched iterator.
        This makes instances of the class directly iterable.
        """
        # --- Handle TensorFlow Dataset ---
        batch_count = 0
        
        try:
            import tensorflow as tf
            if isinstance(self.dataset, tf.data.Dataset):
                logging.info("Detected TensorFlow Dataset. Creating batched iterator.")
                dataset = self.dataset
                # Use .map() for efficient preprocessing in TensorFlow
                if self.preprocess_fn and not (self.preprocess_fn.__code__ is (lambda x:x).__code__):
                    dataset = dataset.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

                for batch in dataset.batch(self.batch_size):
                    # Convert the batch tensor to a list of Python-native types
                    py_batch = []
                    for item in batch:
                        np_item = item.numpy()
                        if isinstance(np_item, bytes):
                            py_batch.append(np_item.decode('utf-8'))
                        else:
                            py_batch.append(np_item)
                    yield py_batch
                    batch_count += 1
                    if self.first_n_batches is not None and batch_count >= self.first_n_batches:
                        break
                return
        except ImportError:
            pass  # TensorFlow not installed

        # --- Handle list, torch.utils.data.Dataset, and datasets.Dataset ---
        items_iterator = None
        
        if isinstance(self.dataset, list):
            items_iterator = iter(self.dataset)
        else:
            try:
                import torch
                if isinstance(self.dataset, torch.utils.data.Dataset):
                    logging.info("Detected PyTorch Dataset.")
                    items_iterator = (self.dataset[i] for i in range(len(self.dataset)))
            except ImportError:
                pass

            try:
                import datasets
                if isinstance(self.dataset, datasets.Dataset):
                    logging.info("Detected Hugging Face Dataset.")
                    items_iterator = iter(self.dataset)
            except ImportError:
                pass
        
        if items_iterator:
            batch = []
            for item in items_iterator:
                processed_item = self.preprocess_fn(item)
                batch.append(processed_item)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    batch_count += 1
                    if self.first_n_batches is not None and batch_count >= self.first_n_batches:
                        break
            if batch: # if the last batch is not full, yield it
                yield batch
            return

        raise TypeError(
            f"Unsupported dataset type: {type(self.dataset).__name__}. "
            "Supported types are: list, torch.utils.data.Dataset, "
            "datasets.Dataset, tf.data.Dataset."
        )
        
    def __len__(self):
        return len(self.dataset)


# --- Generic Data Parallelism Class ---

class DataParallelWorker:
    """
    An abstract base class for a worker in the data parallel pipeline.
    """
    def __init__(self, worker_config: WorkerConfig, **kwargs):
        self.worker_config = worker_config
        self.model = None

    def setup(self):
        raise NotImplementedError("Each worker must implement its own setup method.")

    def process_batch(self, batch: List[Any]) -> List[Any]:
        raise NotImplementedError("Each worker must implement its own process_batch method.")

    def cleanup(self):
        pass

class DataParallelPipeline:
    """
    A general-purpose class for running data-parallel tasks.
    """
    def __init__(self,
                 worker_class: Type[DataParallelWorker],
                 pipeline_config: PipelineConfig,
                 worker_config: WorkerConfig):
        # Check multiprocessing start method
        self._check_multiprocessing_start_method()
        
        if pipeline_config.num_gpus % pipeline_config.gpus_per_worker != 0:
            raise ValueError(
                f"Total GPUs ({pipeline_config.num_gpus}) must be divisible by "
                f"gpus_per_worker ({pipeline_config.gpus_per_worker})."
            )

        self.worker_class = worker_class
        self.pipeline_config = pipeline_config
        self.worker_config = worker_config
        self.num_workers = self.pipeline_config.num_gpus // self.pipeline_config.gpus_per_worker
        effective_queue_size = self.pipeline_config.max_queue_size if self.pipeline_config.max_queue_size > 0 else self.num_workers * 5

        self.task_queue = mp.Queue(maxsize=effective_queue_size)
        self.result_queue = mp.Queue()
        self.setup_queue = mp.Queue()
        self.processes = []

        logging.info(
            f"Pipeline initialized with {self.num_workers} workers, using "
            f"{self.pipeline_config.gpus_per_worker} GPUs per worker. "
            f"Task queue size: {effective_queue_size} batches."
        )

    @staticmethod
    def _check_multiprocessing_start_method():
        """
        Check if multiprocessing start method is set to 'spawn'.
        If not, automatically set it and warn the user.
        """
        current_method = mp.get_start_method(allow_none=True)
        if current_method != 'spawn':
            if current_method is None:
                # No start method set yet, we can set it
                mp.set_start_method('spawn', force=True)
                logging.warning(
                    "Multiprocessing start method was not set. "
                    "Automatically set to 'spawn' for proper GPU handling. "
                    "It's recommended to set this explicitly before initializing DataParallelPipeline."
                )
            else:
                # Start method is already set to something else
                error_msg = (
                    f"Multiprocessing start method is set to '{current_method}' but must be 'spawn'. "
                    f"Please set it using: mp.set_start_method('spawn', force=True) "
                    f"before initializing DataParallelPipeline. "
                    f"This is required for proper GPU handling in multiprocessing."
                )
                raise RuntimeError(error_msg)

    @staticmethod
    def _worker_process_loop(
        worker_index: int, 
        num_workers: int,
        dp_master_port: int,
        dp_master_ip: str,
        gpus_per_worker: int, 
        num_gpus: int,
        worker_class: Type[DataParallelWorker],
        worker_config: WorkerConfig, 
        task_queue: mp.Queue, 
        result_queue: mp.Queue,
        setup_queue: mp.Queue, 
        fetching_task_timeout: float,
        main_log_format: str = None,
        main_log_datefmt: str = None,
        main_log_level: int = None
    ):
        
        # Set up worker logging using custom formatter approach
        DataParallelPipeline._setup_worker_logging(
            worker_index, main_log_format, main_log_datefmt, main_log_level
        )
                
        process_name = f"Worker-{worker_index}"
        mp.current_process().name = process_name
        start_gpu_id = worker_index * gpus_per_worker
        gpu_ids_for_worker = list(range(start_gpu_id, start_gpu_id + gpus_per_worker))
        
        
        
        # ==== for default method, this works
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids_for_worker)) # set the visible GPUs for the worker
        
        logging.info(f"Worker {worker_index} started, assigned to physical GPUs: {gpu_ids_for_worker}.")

        worker = None
        try:
            torch.cuda.empty_cache()
            worker = worker_class(worker_config, **dict(
                num_gpus=num_gpus,
                num_workers=num_workers,
                dp_master_port=dp_master_port,
                dp_master_ip=dp_master_ip,
                gpus_per_worker=gpus_per_worker,
                worker_index=worker_index,
            )) # pass the worker config to the worker, and with the rest of the kwargs for configuring the Data parallel pipeline (optional)
            worker.setup()
            torch.cuda.synchronize()
            setup_queue.put(worker_index)
            
            batch_idx = -1
            while True:
                try:
                    logging.info(f"Waiting for next task to arrive in the task queue...")
                    task = task_queue.get(timeout=fetching_task_timeout)
                    if task is None:
                        logging.info(f"Received shutdown signal. Exiting.")
                        break
                    batch_idx, batch_data, batch_args = task
                    logging.info(f"Received batch {batch_idx} with {len(batch_data)} items for processing.")
                    start_time = time.time()
                    results = worker.process_batch(batch_data)
                    duration = time.time() - start_time
                    
                    # post process results if needed
                    if batch_args.get('batch_size', len(results)) < len(results):
                        results = results[:batch_args['batch_size']]  # trim results to batch size
                    logging.info(f"processed batch {batch_idx} in {duration:.2f}s.")
                    result_queue.put((batch_idx, worker_index, results))
                except Empty:
                    logging.warning(f"timed out waiting for a task.")
                    continue
                except Exception as e:
                    logging.error(f"Error in processing batch: {e}", exc_info=True)
                    if 'batch_idx' in locals():
                        result_queue.put((batch_idx, worker_index, None))
        except Exception as e:
            logging.error(f"Failed to initialize or run worker: {e}", exc_info=True)
        finally:
            if worker: worker.cleanup()
            torch.cuda.empty_cache()
            logging.info(f"Process finished.")

    @staticmethod
    def _setup_worker_logging(worker_index: int, main_log_format: str = None, main_log_datefmt: str = None, main_log_level: int = None):
        """
        Set up worker logging using custom formatter to inherit main process configuration.
        This approach uses a custom formatter that adds worker index to log messages.
        """
        # Use main process format or default
        if main_log_format is None:
            main_log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s"
        
        # Create custom format that includes worker index
        worker_format = main_log_format.replace("%(levelname)s", "%(levelname)s - Worker:%(worker_index)s")
        
        # Create custom formatter
        formatter = WorkerLogFormatter(worker_index, worker_format, main_log_datefmt)
        
        # Set up logging
        logging.basicConfig(
            level=main_log_level if main_log_level is not None else logging.INFO,
            format=worker_format,
            datefmt=main_log_datefmt if main_log_datefmt is not None else "%Y-%m-%d %H:%M:%S",
        )
        
        # Apply the custom formatter to the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    def run(self, 
            data_iterator: Iterator[List[Any]],
            stream_results: bool = False, 
            auto_batch_fill: bool = True
            ) -> Generator[Any, None, None]:
        
        try: 
            # Check if we should run in debug mode (single process)
            if self.pipeline_config.num_gpus == self.pipeline_config.gpus_per_worker:
                logging.info("Debug mode detected: num_gpus == gpus_per_worker. Running in single process mode.")
                yield from self.single_process_run(data_iterator, stream_results)
                return
            
            # Multiprocessing mode if num_gpus > gpus_per_worker
            dp_master_port = get_open_port()
            dp_master_ip = "127.0.0.1"
            
            # Capture main process logging configuration
            root_logger = logging.getLogger()
            main_log_format = None
            main_log_datefmt = None
            main_log_level = root_logger.level
            
            # Try to get the format from the root logger's handlers
            for handler in root_logger.handlers:
                if hasattr(handler, 'formatter') and handler.formatter:
                    main_log_format = handler.formatter._fmt
                    main_log_datefmt = handler.formatter.datefmt
                    break
            
            for worker_index in range(self.num_workers):
                p = mp.Process(target=self._worker_process_loop, args=(
                    worker_index, self.num_workers, dp_master_port, dp_master_ip, self.pipeline_config.gpus_per_worker, self.pipeline_config.num_gpus, self.worker_class,
                    self.worker_config, self.task_queue, self.result_queue,
                    self.setup_queue, self.pipeline_config.fetching_task_timeout,
                    main_log_format, main_log_datefmt, main_log_level
                ))
                p.start()
                self.processes.append(p)
                
                if self.pipeline_config.sequential_worker_setup:
                # NOTE: We should not wait for the first worker to setup, because VLLM will automatically handle that. If we do, the process will hang.
                # === Alternative: Wait for the first worker to setup, not working for VLLM === #
                    try:
                        completed_worker = self.setup_queue.get(timeout=self.pipeline_config.worker_setup_timeout)
                        logging.info(f"Worker {completed_worker} setup completed.")
                    except Empty:
                        raise RuntimeError(f"Worker {worker_index} failed to set up in time.")
                else:
                    time.sleep(1)

            # submit jobs to the task queue
            logging.info(f"All {self.num_workers} workers started. Submitting batches to the task queue...")
            num_batches_submitted = 0
            batch_size = None
            for batch_idx, batch in enumerate(data_iterator):
                
                # get the batch size
                if batch_idx == 0 and batch:
                    batch_size = len(batch)
                    if auto_batch_fill:
                        logging.info(f"First batch size: {batch_size}, auto-filling remaining batches to match this size.")
                        
                # auto batch fill logic
                if auto_batch_fill and batch_size is not None:
                    actual_batch_size = len(batch)
                    if actual_batch_size < batch_size:
                        test_sample = batch[0]
                        batch += [test_sample] * (batch_size - len(batch))  # fill the batch with the first sample to match the size
                        
                        
                self.task_queue.put((batch_idx, batch, dict(
                    batch_size=actual_batch_size if auto_batch_fill else len(batch),
                )))
                num_batches_submitted += 1
            for _ in range(self.num_workers): self.task_queue.put(None)  # send shutdown signal to workers
            logging.info(f"All {num_batches_submitted} batches submitted.")

            num_batches_processed = 0
            while num_batches_processed < num_batches_submitted:
                try:
                    _batch_idx, _worker_index, results = self.result_queue.get(timeout=self.pipeline_config.fetching_result_timeout)
                    num_batches_processed += 1
                    logging.info("Received results for batch %d from worker %d.", _batch_idx, _worker_index)
                    if results is not None: 
                        if stream_results: yield from results # stream the results to the caller, no batch structure
                        else: yield (_batch_idx, results) # return the results in a batch structure
                    else: logging.error(f"Batch {_batch_idx} from worker {_worker_index} failed.")
                except Empty:
                    logging.error("Timeout waiting for results.")
                    break
                
            logging.info(f"All {num_batches_processed} batches processed.")
        
        except Exception as e:
            logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()

    def single_process_run(self, data_iterator: Iterator[List[Any]], stream_results: bool = False) -> Generator[Any, None, None]:
        """
        Run the pipeline in single process mode for debugging.
        This bypasses multiprocessing and runs everything in the main process.
        """
        logging.info("Starting single process run...")
        logging.info(f"Data iterator length: {len(data_iterator)}")
        logging.info(f"stream_results: {stream_results}")
        # Initialize worker in the main process
        worker = None
        try:
            torch.cuda.empty_cache()
            worker = self.worker_class(self.worker_config, **dict(
                num_gpus=self.pipeline_config.num_gpus,
                num_workers=1,  # Single worker in debug mode
                dp_master_port=0,  # Not used in single process mode
                dp_master_ip="127.0.0.1",
                gpus_per_worker=self.pipeline_config.gpus_per_worker,
                worker_index=0,  # Single worker has index 0
            ))
            
            logging.info("Setting up worker...")
            worker.setup()
            torch.cuda.synchronize()
            logging.info("Worker setup completed.")

            # Process batches directly
            for batch_idx, batch_data in enumerate(data_iterator):
                start_time = time.time()
                try:
                    results = worker.process_batch(batch_data)
                    duration = time.time() - start_time
                    logging.info(f"Processed batch {batch_idx} in {duration:.2f}s.")
                    
                    if results is not None:
                        if stream_results:
                            yield from results  # stream the results to the caller, no batch structure
                        else:
                            yield (batch_idx, results)  # return the results in a batch structure
                    else:
                        logging.error(f"Batch {batch_idx} failed.")
                        
                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                    if not stream_results:
                        yield (batch_idx, None)
                        
        except Exception as e:
            logging.error(f"Failed to initialize or run worker: {e}", exc_info=True)
        finally:
            if worker:
                logging.info("Cleaning up worker...")
                worker.cleanup()
            torch.cuda.empty_cache()
            logging.info("Single process run completed.")

    def shutdown(self):
        logging.info("Shutting down dpp pipeline...")
        
        for p in self.processes:
            p.join(timeout=60)
            if p.is_alive():
                logging.warning(f"Process {p.pid} did not terminate gracefully. Forcing.")
                p.terminate()
        self.task_queue.close()
        self.result_queue.close()
        self.setup_queue.close()
        logging.info("DPP pipeline shutdown complete.")

# --- Custom Logging Formatter for Workers ---
class WorkerLogFormatter(logging.Formatter):
    """Custom formatter that adds worker index to log messages."""
    
    def __init__(self, worker_index: int, fmt=None, datefmt=None, style='%'):
        self.worker_index = worker_index
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        # Add worker index to the record
        record.worker_index = self.worker_index
        return super().format(record)
