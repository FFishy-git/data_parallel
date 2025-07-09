import sys, os 
# add parent directory to path using getcwd
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
def add_parent_dir_to_sys_path(parent_depth: int = 1):
    """
    Adds the parent directory to sys.path to allow importing from the parent directory.
    This is useful for running scripts that are not in the same directory as the module.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        current_dir = os.getcwd()
    for _ in range(parent_depth):
        current_dir = os.path.dirname(current_dir)
    parent_dir = current_dir
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)


add_parent_dir_to_sys_path(2) # Ensure parent directory is in sys.path


import multiprocessing as mp
import glob, os
from datasets import load_dataset, Dataset
from data_parallel.lib.data_parallel.dpp import DataIterator
from vllm_dpp_eval  import VLLM_DPP_Evalutation_FrameWork  # wherever you defined it
import logging

# 1) Configure logging before anything else uses it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logging.info("Starting C2S vLLM DPP inference")

if __name__ == "__main__":
    

    # Redirect HF cache to scratch
    # redirect all HF & Datasets caches into your home scratch
    user = os.getenv("USER")

    # where you have write permission
    hf_cache_dir       = f"/home/{user}/scratch/bio_transcoder_data/hf_cache"
    hf_datasets_cache  = f"/home/{user}/scratch/bio_transcoder_data/hf_datasets_cache"

    # create them if they don’t exist
    os.makedirs(hf_cache_dir,      exist_ok=True)
    os.makedirs(hf_datasets_cache, exist_ok=True)

    # point Transformers and Datasets at these
    os.environ["HF_HOME"]             = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"]  = hf_cache_dir
    os.environ["HF_DATASETS_CACHE"]   = hf_datasets_cache

    mp.set_start_method("spawn", force=True)

    # 1) Load your C2S Arrow split
    data_dir = "/home/hs925/scratch/bio_transcoder_data/dominguez_immune_tissue_multi_task_samples_HF_ds"
    split   = "test"
    ds      = load_dataset("arrow", data_files=f"{data_dir}/{split}/*.arrow",
                           cache_dir=hf_datasets_cache)["train"]

    # 2) Extract prompts
    preprocess_fn = lambda ex: ex["model_input"]

    # 3) Instantiate framework
    framework = VLLM_DPP_Evalutation_FrameWork(
        model_id="vandijklab/C2S-Scale-Pythia-1b-pt",
        max_new_tokens=1024,
        temperature=0.0,
        batch_size=64,
        gpus_per_worker=1,
        max_queue_size=128,
        first_n_batches=None,
    )

    # 4) Run it!
    results = framework(dataset=ds, preprocess_fn=preprocess_fn)
    # `results` is already a list of dicts:
    #   [{"prompt": ..., "generated_text": ...}, {...}, ...]

    # 5) Just save `results` directly
    out_ds = Dataset.from_list(results)
    out_dir = "/gpfs/radev/project/zhuoran_yang/hs925/Bio_Transcoder/dataset_prepare/bio_data_check/generated_data/test"
    os.makedirs(out_dir, exist_ok=True)
    out_ds.save_to_disk(out_dir)

    print(f"Saved {len(results)} records to {out_dir}")
