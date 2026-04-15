"""
vLLM Client for Open-Source LLMs

Paper models (Section 4.1, Appendix A.5):
  DeepSeek-R1-Distill-Qwen-14B, Gemma 3 12B, Ministral-3-14B, Qwen 2.5 14B
"""

import os
import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Try importing vLLM (for direct inference)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Using OpenAI-compatible API mode.")

# Try importing OpenAI client (for vLLM server mode)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VLLMClient:
    """
    Client for open-source LLMs using vLLM.
    
    Supports two modes:
    1. Direct mode: Load model directly using vLLM
    2. Server mode: Connect to vLLM server via OpenAI-compatible API
    """
    
    # Model-specific chat templates
    CHAT_TEMPLATES = {
        "qwen": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        "deepseek": "<|begin▁of▁sentence|>User: {user}\n\nAssistant: ",
        "gemma": "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n",
        "mistral": "[INST] {user} [/INST]",
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        model_path: Optional[str] = None,
        download_dir: Optional[str] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        gpu_memory_utilization: float = 0.70,
        server_url: Optional[str] = None,
        use_server: bool = False,
    ):
        """
        Initialize vLLM client.
        
        Args:
            model_name: HuggingFace model name (used if model_path is None)
            model_path: Local path to model weights (takes precedence over model_name)
            download_dir: Directory where models are downloaded/cached
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type (bfloat16, float16)
            max_model_len: Maximum context length
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: Fraction of GPU memory to use
            server_url: URL of vLLM server (for server mode)
            use_server: Whether to use server mode
        """
        # Use local path if provided, otherwise use HuggingFace model name
        self.model_path = model_path or model_name
        self.model_name = model_name  # Keep original name for reference
        self.download_dir = download_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_url = server_url or "http://localhost:8000/v1"
        self.use_server = use_server
        
        # Determine model family for chat template
        self.model_family = self._get_model_family(model_name)
        
        # Initialize based on mode
        if use_server:
            self._init_server_mode()
        else:
            self._init_direct_mode()
        
        logger.info(f"Initialized vLLM client for {model_name} (family: {self.model_family})")
    
    def _get_model_family(self, model_name: str) -> str:
        """Determine model family from name."""
        model_lower = model_name.lower()
        if "deepseek" in model_lower:
            return "deepseek"
        elif "gemma" in model_lower:
            return "gemma"
        elif "mistral" in model_lower or "ministral" in model_lower:
            return "mistral"
        elif "qwen" in model_lower:
            return "qwen"
        else:
            return "qwen"  # Default
    
    def _init_direct_mode(self):
        """Initialize direct vLLM inference."""
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Or use server mode by setting use_server=True"
            )
        
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")
        if self.download_dir:
            logger.info(f"Download directory: {self.download_dir}")
        
        # Build kwargs for LLM initialization
        llm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        
        # Add download_dir if specified
        if self.download_dir:
            llm_kwargs["download_dir"] = self.download_dir
        
        self.llm = LLM(**llm_kwargs)
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        self.client = None
    
    def _init_server_mode(self):
        """Initialize OpenAI-compatible client for vLLM server."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for server mode")
        
        self.client = OpenAI(
            base_url=self.server_url,
            api_key="EMPTY",  # vLLM doesn't need real key
        )
        self.llm = None
        self.sampling_params = None
        
        logger.info(f"Using vLLM server at {self.server_url}")
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt using appropriate chat template."""
        system = system_prompt or "You are a helpful assistant."

        template = self.CHAT_TEMPLATES.get(self.model_family, self.CHAT_TEMPLATES["qwen"])

        if self.model_family in ("deepseek", "gemma", "mistral"):
            return template.format(user=prompt)
        else:
            return template.format(system=system, user=prompt)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.use_server and self.client:
            return self._generate_server(prompt, system_prompt, **kwargs)
        else:
            return self._generate_direct(prompt, system_prompt, **kwargs)
    
    def _generate_direct(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using direct vLLM inference."""
        # Update sampling params if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Format prompt
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        # Generate
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        return outputs[0].outputs[0].text
    
    def _generate_server(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using vLLM server (OpenAI-compatible API)."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Use model_path for server mode (vLLM uses local path as model ID)
        # This handles both local paths and HuggingFace names
        model_id = self.model_path
        
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"vLLM server error: {e}")
            raise
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        if self.use_server:
            # Server mode: sequential generation
            return [self.generate(p, system_prompt, **kwargs) for p in prompts]
        else:
            # Direct mode: batch generation
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            formatted_prompts = [
                self._format_prompt(p, system_prompt) for p in prompts
            ]
            
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            
            return [output.outputs[0].text for output in outputs]


# Default model directory (update to your local path)
DEFAULT_MODEL_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Paper models (Section 4.1, Appendix A.5): four 12-14B instruction-tuned models
MODEL_CONFIGS = {
    "deepseek_r1_14b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "local_dir": "DeepSeek-R1-Distill-Qwen-14B",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    "gemma_3_12b": {
        "model_name": "google/gemma-3-12b-it",
        "local_dir": "gemma-3-12b-it",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    "ministral_14b": {
        "model_name": "mistralai/Ministral-3-14B-Instruct-2512",
        "local_dir": "Ministral-3-14B-Instruct-2512",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
    "qwen_2_5_14b": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "local_dir": "Qwen2.5-14B-Instruct",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
    },
}

class Qwen25_14BClient(VLLMClient):
    """Pre-configured client for Qwen 2.5 14B Instruct."""

    def __init__(self, model_dir: Optional[str] = None, **kwargs):
        config = MODEL_CONFIGS["qwen_2_5_14b"]
        model_dir = model_dir or DEFAULT_MODEL_DIR
        local_path = os.path.join(model_dir, config["local_dir"])

        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = config["model_name"]

        kwargs.setdefault("tensor_parallel_size", 1)
        kwargs.setdefault("max_model_len", 8192)
        kwargs.pop("model_name", None)
        kwargs.pop("model_path", None)

        super().__init__(
            model_name=config["model_name"],
            model_path=model_path,
            download_dir=model_dir,
            **kwargs
        )
        self.name = "qwen_2_5_14b"


class DeepSeekR1_14BClient(VLLMClient):
    """Pre-configured client for DeepSeek R1 Distill Qwen 14B (Jan 2025)."""
    
    def __init__(self, model_dir: Optional[str] = None, **kwargs):
        config = MODEL_CONFIGS["deepseek_r1_14b"]
        model_dir = model_dir or DEFAULT_MODEL_DIR
        local_path = os.path.join(model_dir, config["local_dir"])
        
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = config["model_name"]
        
        kwargs.setdefault("tensor_parallel_size", 1)
        kwargs.setdefault("max_model_len", 8192)
        
        kwargs.pop("model_name", None)
        kwargs.pop("model_path", None)
        
        super().__init__(
            model_name=config["model_name"],
            model_path=model_path,
            download_dir=model_dir,
            **kwargs
        )
        self.name = "deepseek_r1_14b"


class Ministral14BClient(VLLMClient):
    """Pre-configured client for Mistral Ministral 3 14B (Dec 2024)."""
    
    def __init__(self, model_dir: Optional[str] = None, **kwargs):
        config = MODEL_CONFIGS["ministral_14b"]
        model_dir = model_dir or DEFAULT_MODEL_DIR
        local_path = os.path.join(model_dir, config["local_dir"])
        
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = config["model_name"]
        
        kwargs.setdefault("tensor_parallel_size", 1)
        kwargs.setdefault("max_model_len", 8192)
        
        kwargs.pop("model_name", None)
        kwargs.pop("model_path", None)
        
        super().__init__(
            model_name=config["model_name"],
            model_path=model_path,
            download_dir=model_dir,
            **kwargs
        )
        self.name = "ministral_14b"


class Gemma3_12BClient(VLLMClient):
    """Pre-configured client for Google Gemma 3 12B (Mar 2025)."""
    
    def __init__(self, model_dir: Optional[str] = None, **kwargs):
        config = MODEL_CONFIGS["gemma_3_12b"]
        model_dir = model_dir or DEFAULT_MODEL_DIR
        local_path = os.path.join(model_dir, config["local_dir"])
        
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = config["model_name"]
        
        kwargs.setdefault("tensor_parallel_size", 1)
        kwargs.setdefault("max_model_len", 8192)
        
        kwargs.pop("model_name", None)
        kwargs.pop("model_path", None)
        
        super().__init__(
            model_name=config["model_name"],
            model_path=model_path,
            download_dir=model_dir,
            **kwargs
        )
        self.name = "gemma_3_12b"

