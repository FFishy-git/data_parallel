from dataclasses import dataclass
from typing import List, Dict
import logging
import multiprocessing as mp
from lib.data_parallel.dpp import DataParallelPipeline, DataParallelWorker, WorkerConfig, PipelineConfig
from vllm import LLM, SamplingParams
import os

# --- Example Implementation with Structured Config ---

@dataclass
class VLLMWorkerConfig(WorkerConfig):
    model_id: str
    tokenizer: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1 # -1 means all tokens are considered
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    enforce_eager: bool = True
    # Additional VLLM compatibility options
    dtype: str = "auto"  # auto, float16, bfloat16, float32
    max_model_len: int = 8192
    disable_log_stats: bool = False
    max_num_batched_tokens: int = 4096

class VLLMInferenceWorker(DataParallelWorker):
    def __init__(self, 
                 worker_config: VLLMWorkerConfig, 
                 num_gpus: int,
                 num_workers: int,
                 dp_master_port: int,
                 dp_master_ip: str,
                 gpus_per_worker: int,
                 worker_index: int,
                 ):
        super().__init__(worker_config)
        self.sampling_params = None
        
        self.gpus_per_worker = gpus_per_worker
        
        # Log the worker config during initialization
        logging.info(f"VLLMWorkerConfig: {self.worker_config}")
        
        # ==== for VLLM, we need to set all the GPUs visible to the worker
        os.environ["VLLM_DP_RANK"] = str(worker_index) # We currently only support 1 global rank
        # os.environ["VLLM_DP_RANK_LOCAL"] = str(worker_index)
        os.environ["VLLM_DP_SIZE"] = str(num_workers)
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        
        # set the visible GPUs for the worker
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        logging.info(f"Environment variables set\n"
                     f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'Not set'}\n"
                     f"VLLM_DP_RANK: {os.environ['VLLM_DP_RANK']}\n"
                     f"VLLM_DP_RANK_LOCAL: {os.environ['VLLM_DP_RANK_LOCAL'] if 'VLLM_DP_RANK_LOCAL' in os.environ else 'Not set'}\n"
                     f"VLLM_DP_SIZE: {os.environ['VLLM_DP_SIZE']}\n"
                     f"VLLM_DP_MASTER_PORT: {os.environ['VLLM_DP_MASTER_PORT']}\n"
                     f"VLLM_DP_MASTER_IP: {os.environ['VLLM_DP_MASTER_IP']}")
        
    def setup(self):
        tensor_parallel_size = self.gpus_per_worker
        
        # Add additional VLLM options to handle Gemma3 compatibility issues
        vllm_kwargs = {
            "model": self.worker_config.model_id,
            # "tokenizer": self.worker_config.tokenizer if hasattr(self.worker_config, "tokenizer") else self.worker_config.model_id,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": self.worker_config.gpu_memory_utilization,
            "enforce_eager": self.worker_config.enforce_eager,
            "trust_remote_code": self.worker_config.trust_remote_code,
            "dtype": self.worker_config.dtype,
            "max_model_len": self.worker_config.max_model_len,
            "max_num_batched_tokens": self.worker_config.max_num_batched_tokens,
        }
        
        # Add compatibility options for Gemma3 models
        if "gemma" in self.worker_config.model_id.lower():
            vllm_kwargs.update({
                "dtype": "bfloat16",  # Use bfloat16 for better compatibility
                "max_model_len": 8192,  # Limit model length
                "disable_log_stats": True,  # Disable some logging that might cause issues
            })
            logging.info("Added Gemma-specific VLLM compatibility options")
        
        if self.worker_config.disable_log_stats:
            vllm_kwargs["disable_log_stats"] = True
        
        logging.info(f"Initializing VLLM model with kwargs: {vllm_kwargs}")
        
        try:
            self.model = LLM(**vllm_kwargs)
            logging.info("VLLM model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize VLLM model: {e}")
            # Try with more conservative settings
            logging.info("Retrying with more conservative settings...")
            vllm_kwargs.update({
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.7,
                "max_num_batched_tokens": 2048,
            })
            self.model = LLM(**vllm_kwargs)
            logging.info("VLLM model initialized with conservative settings")
        
        self.sampling_params = SamplingParams(
            temperature=self.worker_config.temperature,
            top_p=self.worker_config.top_p,
            top_k=self.worker_config.top_k,
            max_tokens=self.worker_config.max_new_tokens,
        )
        logging.info("Model and sampling params initialized.")

    def process_batch(self, batch: List[str]) -> List[Dict[str, str]]:
        logging.debug(f"Processing batch: {batch}")
        try:
            outputs = self.model.generate(batch, self.sampling_params, use_tqdm=False)
            logging.debug(f"model.generate() outputs: {outputs}")
            # Normal case: outputs are objects with attributes
            return [{"prompt": o.prompt, "generated_text": o.outputs[0].text} for o in outputs]
        except Exception as e:
            logging.error(f"Failed to generate outputs for batch: {e}")
            # Fallback case: create empty outputs as dictionaries
            fallback_outputs = [{"prompt": p, "outputs": [{"text": ""}]} for p in batch]
            logging.debug(f"Using fallback outputs: {fallback_outputs}")
            # Handle fallback outputs as dictionaries
            return [{"prompt": o["prompt"], "generated_text": o["outputs"][0]["text"]} for o in fallback_outputs]

if __name__ == "__main__":
    from lib.data_parallel.dpp import DataIterator
    
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mp.set_start_method("spawn", force=True)

    # 1. Configure Pipeline and Worker
    pipeline_config = PipelineConfig(gpus_per_worker=1)
    worker_config = VLLMWorkerConfig(
        model_id="google/gemma-2b-it",
        max_new_tokens=64,
        temperature=0.1
    )

    # 2. Prepare Data and Preprocessing
    sample_data_dicts = [{"prompt": f"Describe {topic} in one sentence.", "id": i} for i, topic in enumerate([
        "the Roman Empire", "dark matter", "blockchain", "the theory of relativity",
        "the Amazon rainforest", "neural networks", "the Renaissance", "plate tectonics",
    ])] * 5
    def extract_prompt(item: dict) -> str:
        return item['prompt']

    # 3. Create the data iterator using the new DataIteratorFactory class
    num_workers = pipeline_config.num_gpus // pipeline_config.gpus_per_worker
    batch_size = 8 * num_workers
    data_iterator = DataIterator(
        dataset=sample_data_dicts,
        batch_size=batch_size,
        preprocess_fn=extract_prompt
    )

    # 4. Initialize and Run Pipeline
    pipeline = DataParallelPipeline(
        worker_class=VLLMInferenceWorker,
        pipeline_config=pipeline_config,
        worker_config=worker_config,
    )

    try:
        all_results = []
        for result in pipeline.run(data_iterator=data_iterator):
            all_results.append(result)
            print(f"Result for '{result['prompt']}': '{result['generated_text']}'")
        print(f"\nTotal results collected: {len(all_results)}")
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        pipeline.shutdown()
