import os
import subprocess
from llama_cpp import Llama
from openai import OpenAI
from loguru import logger

GLOBAL_LLM = None

# Repo of the local GGUF model.
_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
# Qwen2.5-3B q4_k_m needs ~2.5GB; require some headroom for KV cache / compute buffers.
_MIN_FREE_MIB = 4096 * 2


def _query_gpus() -> list[tuple[int, int]]:
    """Return list of (gpu_index, free_memory_MiB), sorted by free memory desc.

    Returns an empty list if nvidia-smi is unavailable or fails (e.g. no GPU).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        logger.warning(f"Cannot query GPUs via nvidia-smi ({e}).")
        return []
    gpus = []
    for line in out.strip().splitlines():
        try:
            idx, free = (x.strip() for x in line.split(","))
            gpus.append((int(idx), int(free)))
        except ValueError:
            continue
    gpus.sort(key=lambda x: x[1], reverse=True)
    return gpus


def _load_on_gpu() -> Llama:
    """Load the model on the GPU(s) visible via CUDA_VISIBLE_DEVICES. Raises on failure."""
    return Llama.from_pretrained(
        repo_id=_MODEL_REPO,
        filename=_MODEL_FILE,
        n_ctx=5_000,
        n_gpu_layers=-1,
        main_gpu=0,
        n_threads=4,
        verbose=False,
    )


def _load_on_cpu() -> Llama:
    logger.info("Loading local LLM on CPU.")
    return Llama.from_pretrained(
        repo_id=_MODEL_REPO,
        filename=_MODEL_FILE,
        n_ctx=5_000,
        n_gpu_layers=0,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )


def _load_local_llm() -> Llama:
    # run.sh is responsible for pinning CUDA_VISIBLE_DEVICES to a single free GPU
    # (or "" for CPU) BEFORE this process starts, so that both torch and llama-cpp
    # only ever see one card and never split the model. We respect that decision
    # here and only re-derive it if the variable was not set (e.g. direct python run).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cvd is None:
        # Not launched via run.sh: pick the most-free GPU ourselves.
        gpus = _query_gpus()
        candidates = [(idx, free) for idx, free in gpus if free >= _MIN_FREE_MIB]
        if candidates:
            idx, free = candidates[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            logger.info(f"Auto-selected GPU {idx} ({free} MiB free).")
            cvd = str(idx)
        else:
            best = max((free for _, free in gpus), default=0)
            logger.warning(
                f"No GPU has >= {_MIN_FREE_MIB} MiB free (max free = {best} MiB); using CPU."
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            cvd = ""

    if cvd.strip() == "":
        return _load_on_cpu()

    try:
        logger.info(f"Loading local LLM on GPU (CUDA_VISIBLE_DEVICES={cvd}).")
        llm = _load_on_gpu()
        logger.info("Local LLM loaded on GPU.")
        return llm
    except Exception as e:
        logger.warning(f"GPU loading failed ({e}); falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return _load_on_cpu()

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = _load_local_llm()
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            response = self.llm.chat.completions.create(messages=messages,temperature=0,model=self.model,max_tokens=1024)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0,max_tokens=1024)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM