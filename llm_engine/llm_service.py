from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch
from .llm_request_handler import handle_query_requests
from groq import Groq
from settings import LLM_SERVICE, LOCAL_MODEL, DEEPSEEK_R1_CONFIG

class LLMServices:
    def __init__(self):
        self.model = None

        if LLM_SERVICE == 'local':
            self.load_llm(LOCAL_MODEL)
        elif LLM_SERVICE == 'groq':
            self.model = Groq()

    def show_vram(self):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        return f"Karen VRAM: alloc={allocated:.2f}GB | reserved={reserved:.2f}GB"
    
    def show_vram_bar(self):
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved  = torch.cuda.memory_reserved(device)  / 1024**3
        total     = torch.cuda.get_device_properties(device).total_memory / 1024**3

        total_cubes     = 24
        alloc_cubes     = int(total_cubes * allocated / total)
        reserved_cubes  = int(total_cubes * reserved  / total)

        bar = (
            "▮" * alloc_cubes +
            "▬" * (reserved_cubes - alloc_cubes) +
            "▯" * (total_cubes - reserved_cubes)
        )
        return f"VRAM: {allocated:.2f} alloc / {reserved:.2f} reserved / {total:.2f}GB total [{bar}]"


    def get_llm_response(self, request):
        response = handle_query_requests(request, self.model)
        if LLM_SERVICE == 'local':
            print(self.show_vram_bar())
        return response

    def load_llm(self, llm_model_name = 'distill-qwen'):

        if llm_model_name == 'distill-qwen':
            quantize_model = '4-bit'

            print("LOADING QWEN...")

            # Setup quantization config
            quantization_config = None
            if quantize_model == '4-bit':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            
            elif quantize_model == '8bit':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0)

            # Load tokenizer
            autotokenizer = AutoTokenizer.from_pretrained(
                DEEPSEEK_R1_CONFIG['model_path'],
                trust_remote_code=True,
                local_files_only=True
            )

            # Ensure padding token is set
            if autotokenizer.pad_token is None:
                autotokenizer.pad_token = autotokenizer.eos_token

            # Load model with quantization
            automodel = AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_R1_CONFIG['model_path'],
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
                quantization_config=quantization_config
            )

            # Create pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=automodel,
                tokenizer=autotokenizer,
                max_new_tokens=DEEPSEEK_R1_CONFIG['max_new_tokens'],
                temperature=DEEPSEEK_R1_CONFIG['temperature'],
                top_p=0.5,
                top_k=50,
                repetition_penalty=DEEPSEEK_R1_CONFIG['repeat_penalty'],
                do_sample=True,
                return_full_text=False,
                pad_token_id=autotokenizer.pad_token_id,
            )

            # Wrap into LangChain-compatible LLM
            llm = HuggingFacePipeline(
                pipeline=hf_pipeline,
                model_kwargs={
                    "n_ctx": DEEPSEEK_R1_CONFIG['nctx'],
                    "seed": 3667,
                }
            )

        self.model = llm