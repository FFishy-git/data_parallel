from lib.data_parallel.dpp import (
    DataParallelPipeline,
    PipelineConfig,
    DataIterator,
)
from lib.data_parallel.dpp_vllm import VLLMWorkerConfig, VLLMInferenceWorker
import multiprocessing as mp
import logging
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class VLLM_DPP_Evalutation_FrameWork:
    model_id: str
    max_new_tokens: int
    temperature: float
    batch_size: int
    gpus_per_worker: int
    max_queue_size: int
    
    # debug features
    first_n_batches: int | None = None
    
    def __call__(self, dataset, preprocess_fn):
        return self.run(dataset=dataset, preprocess_fn=preprocess_fn)

    def run(self, dataset, preprocess_fn):
        mp.set_start_method("spawn", force=True)

        # 1. Structured Configuration
        pipeline_config = PipelineConfig(
            gpus_per_worker=self.gpus_per_worker,
            max_queue_size=self.max_queue_size,
        )
        worker_config = VLLMWorkerConfig(
            model_id=self.model_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        
        num_workers = pipeline_config.num_gpus // pipeline_config.gpus_per_worker
        batch_size = self.batch_size
        
        # Calculate total number of batches for progress bar
        total_samples = len(dataset) if self.first_n_batches is None else min(self.first_n_batches * batch_size, len(dataset))
        
        # Create data iterator
        data_iterator = DataIterator(
            dataset=dataset, 
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            first_n_batches=self.first_n_batches
        )

        total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

        # 3. Initialize and Run Pipeline
        pipeline = DataParallelPipeline(
            worker_class=VLLMInferenceWorker,
            pipeline_config=pipeline_config,
            worker_config=worker_config,
        )

        try:
            all_results = [[] for _ in range(total_batches)]
            total_samples_processed = 0
            
            # Create progress bar
            with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
                for i, (batch_idx, results) in enumerate(pipeline.run(data_iterator=data_iterator, stream_results=False)):
                    all_results[batch_idx] = results
                    total_samples_processed += len(results) if results else 0
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'returned_batch_idx': i,
                        'batch_size': len(results) if results else 0,
                        'samples_processed': total_samples_processed,
                        'total_samples': total_samples,
                        'progress': f"{total_samples_processed}/{total_samples}"
                    })
                    
                    # Log first batch result for debugging
                    if batch_idx == 0 and results:
                        logging.debug(f"First returned batch result sample: {results[0]}")
                        print(f"Result of first sample: {results[0]}")
                    
                    # # Log every 10th batch for monitoring
                    # if len(all_results) % 10 == 0:
                    #     logging.info(f"Processed {len(all_results)}/{total_batches} batches ({total_samples_processed}/{total_samples} samples)")
            
            logging.info(f"\nTotal results collected: {len(all_results)}")
            logging.info(f"Total samples processed: {total_samples_processed}")
            logging.info(f"Pipeline completed successfully. Total batches: {len(all_results)}, Total samples: {total_samples_processed}")
            
        except Exception as e:
            logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        finally:
            pipeline.shutdown()
            
        # all_results is a list of lists, each list contains the results of a batch. Let us flatten it.
        all_results = [item for sublist in all_results for item in sublist]
        return all_results
            
if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="data/evaluation/data/vllm_dpp_eval.json", split="train")
    preprocess_fn = lambda x: x["text"]
    
    # Use a more compatible model for testing
    # Try gemma-2b-it first as it's more stable with VLLM
    model_id = "google/gemma-2b-it"  # Changed from gemma-3-4b-it to avoid compatibility issues
    
    results = VLLM_DPP_Evalutation_FrameWork(
        model_id=model_id,
        max_new_tokens=1024,
        temperature=0.0,
        batch_size=32,  # Reduced batch size for better stability
        gpus_per_worker=1,
        max_queue_size=128  # Reduced queue size
    )(dataset=dataset, preprocess_fn=preprocess_fn)
    print(results)