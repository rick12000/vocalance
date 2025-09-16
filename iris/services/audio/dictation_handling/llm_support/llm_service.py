"""
Streamlined LLM Service

Simplified LLM processing with streaming support and minimal complexity.
Preserves core functionality while removing over-engineering.
"""
import asyncio
import logging
import os
import threading
import re
from typing import Optional, Callable
from llama_cpp import Llama

from iris.event_bus import EventBus
from iris.config.app_config import GlobalAppConfig
from iris.services.storage.llm_model_downloader import LLMModelDownloader
from iris.events.dictation_events import LLMProcessingCompletedEvent, LLMProcessingFailedEvent

logger = logging.getLogger(__name__)

class LLMService:
    """Streamlined LLM service with essential functionality"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        self.event_bus = event_bus
        self.config = config
        self.llm: Optional[Llama] = None
        self._model_loaded = False
        self._processing_lock = threading.RLock()
        
        # Initialize model downloader
        self.model_downloader = LLMModelDownloader(config)
        
        # Get model configuration
        self.model_info = config.llm.get_model_info()
        self.model_filename = config.llm.get_model_filename()
        self.model_path: Optional[str] = None  # Will be set during initialization
        self.context_length = config.llm.context_length
        self.n_threads = config.llm.n_threads
        
        logger.info(f"LLMService initialized for model: {self.model_filename}")
    
    async def initialize(self) -> bool:
        """Initialize LLM model - download if needed, then load"""
        try:
            # Check if model exists locally, download if not
            if not self.model_downloader.model_exists(self.model_filename):
                logger.info(f"Model {self.model_filename} not found locally, downloading...")
                model_path = await self.model_downloader.download_model(
                    self.model_info["repo_id"], 
                    self.model_info["filename"]
                )
                if not model_path:
                    logger.error("Failed to download model")
                    return False
            else:
                logger.info(f"Model {self.model_filename} found locally")
            
            # Get the model path and store it in instance variable
            self.model_path = self.model_downloader.get_model_path(self.model_filename)
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found after download: {self.model_path}")
                return False
            
            logger.info(f"Loading LLM model: {os.path.basename(self.model_path)}")
            
            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(None, self._load_model, self.model_path)
            
            if self.llm:
                self._model_loaded = True
                logger.info("LLM model loaded successfully")
                return True
            else:
                logger.error("Failed to load LLM model")
                return False
                
        except Exception as e:
            logger.error(f"LLM initialization error: {e}", exc_info=True)
            return False
    
    def _load_model(self, model_path: str) -> Optional[Llama]:
        """Load Llama model with optimized settings"""
        try:
            model = Llama(
                model_path=model_path,
                n_ctx=self.context_length,
                n_threads=self.n_threads,
                verbose=False,
                n_batch=512,
                seed=42,
                f16_kv=True,
                use_mmap=True,
                use_mlock=True,
                n_gpu_layers=0
            )
            return model
        except Exception as e:
            logger.error(f"Model loading error: {e}", exc_info=True)
            return None
    
    async def process_dictation(self, raw_text: str, agentic_prompt: str) -> Optional[str]:
        """Process dictation text with LLM"""
        if not self._model_loaded or not self.llm:
            logger.error("LLM model not loaded")
            return None
        
        with self._processing_lock:
            try:
                logger.info(f"Processing dictation: '{raw_text[:50]}...'")
                
                prompt = self._build_prompt(agentic_prompt, raw_text)
                result = await self._generate(prompt)
                
                if result:
                    cleaned = self._clean_output(result)
                    if self._validate_output(cleaned, raw_text):
                        logger.info("Processing completed successfully")
                        await self._publish_completed(cleaned, agentic_prompt)
                        return cleaned
                
                # Fallback to cleaned original
                logger.warning("Processing failed, returning cleaned original")
                cleaned_original = self._clean_text(raw_text)
                await self._publish_completed(cleaned_original, agentic_prompt)
                return cleaned_original
                
            except Exception as e:
                error_msg = f"LLM processing error: {e}"
                logger.error(error_msg, exc_info=True)
                await self._publish_failed(error_msg, raw_text)
                return self._clean_text(raw_text)
    
    async def process_dictation_streaming(self, raw_text: str, agentic_prompt: str, 
                                        token_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """Process dictation with streaming output"""
        if not self._model_loaded or not self.llm:
            logger.error("LLM model not loaded")
            return None
        
        with self._processing_lock:
            try:
                logger.info(f"Streaming processing: '{raw_text[:50]}...'")
                
                prompt = self._build_prompt(agentic_prompt, raw_text)
                result = await self._generate_streaming(prompt, token_callback)
                
                if result:
                    cleaned = self._clean_output(result)
                    if self._validate_output(cleaned, raw_text):
                        logger.info("✅ Streaming processing completed successfully")
                        await self._publish_completed(cleaned, agentic_prompt)
                        return cleaned
                
                # Fallback
                logger.warning("Streaming failed, returning cleaned original")
                cleaned_original = self._clean_text(raw_text)
                await self._publish_completed(cleaned_original, agentic_prompt)
                return cleaned_original
                
            except Exception as e:
                error_msg = f"Streaming processing error: {e}"
                logger.error(error_msg, exc_info=True)
                await self._publish_failed(error_msg, raw_text)
                return self._clean_text(raw_text)
    
    def _build_prompt(self, agentic_prompt: str, raw_text: str) -> str:
        """Build LLM prompt"""
        system = f"""You are a professional task executer.  You will be given a task and a list of instructions to apply to a text input.  You will need to execute the task based on the instructions.
                    Task and instructions: {agentic_prompt.strip()}"""

        user = f"Text input:\n\n{raw_text.strip()}"
        return self._format_for_model(system, user)
    
    def _format_for_model(self, system: str, user: str) -> str:
        """Format prompt for specific model type"""
        model_lower = self.model_path.lower()
        
        if "qwen" in model_lower:
            return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in model_lower and "chat" in model_lower:
            return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
        elif "mistral" in model_lower:
            return f"<s>[INST] {system}\n\n{user} [/INST]"
        else:
            return f"### System\n{system}\n\n### User\n{user}\n\n### Assistant\n"
    
    async def _generate(self, prompt: str) -> Optional[str]:
        """Generate non-streaming response"""
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._llm_generate, prompt),
                timeout=30.0
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Generation timed out")
            return None
    
    def _llm_generate(self, prompt: str) -> Optional[str]:
        """Execute LLM generation"""
        try:
            # Calculate tokens based on input length
            input_tokens = len(prompt.split()) * 1.3
            max_tokens = max(256, min(800, int(input_tokens * 2)))
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.05,
                stop=["<|im_end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:"],
                echo=False,
                stream=False
            )
            
            if response and response.get('choices'):
                return response['choices'][0]['text'].strip()
            return None
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return None
    
    async def _generate_streaming(self, prompt: str, token_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """Generate streaming response"""
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._llm_generate_streaming, prompt, token_callback),
                timeout=30.0
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Streaming generation timed out")
            return None
    
    def _llm_generate_streaming(self, prompt: str, token_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """Execute streaming LLM generation"""
        try:
            input_tokens = len(prompt.split()) * 1.3
            max_tokens = max(256, min(800, int(input_tokens * 2)))
            
            full_text = ""
            stop_sequences = ["<|im_end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:"]
            
            stream = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.05,
                stop=stop_sequences,
                echo=False,
                stream=True
            )
            
            for chunk in stream:
                if chunk and chunk.get('choices'):
                    token = chunk['choices'][0]['text']
                    if token:
                        full_text += token
                        
                        # Stream token to callback
                        if token_callback:
                            try:
                                token_callback(token)
                            except Exception as e:
                                logger.error(f"Token callback error: {e}", exc_info=True)
                        
                        # Check for early stopping
                        if self._should_stop_generation(full_text, stop_sequences):
                            break
            
            return full_text.strip() if full_text else None
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}", exc_info=True)
            return None
    
    def _should_stop_generation(self, text: str, stop_sequences: list) -> bool:
        """Check if generation should stop early"""
        for seq in stop_sequences:
            if seq in text:
                return True
        
        # Stop if text becomes repetitive
        if len(text) > 100:
            words = text.split()
            if len(words) > 20:
                last_10 = ' '.join(words[-10:])
                prev_10 = ' '.join(words[-20:-10])
                if last_10 == prev_10:
                    return True
        
        return False
    
    def _clean_output(self, text: str) -> str:
        """Clean LLM output while preserving formatting"""
        if not text:
            return ""
        
        # Remove common artifacts
        text = re.sub(r'(?i)^(?:here\'?s?|this is|below is)\s+(?:the\s+)?(?:improved|corrected|fixed|better|revised)\s+(?:text|version)[:.]?\s*', '', text)
        text = re.sub(r'(?i)^as an? (?:ai|assistant|language model).*?[.!?]\s*', '', text)
        text = re.sub(r'(?i)^(?:sure|certainly|of course|absolutely)[,.]?\s*', '', text)
        
        # Clean up formatting while preserving intentional structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines to 2
        text = text.strip()
        
        return text
    
    def _validate_output(self, output: str, original: str) -> bool:
        """Validate output quality"""
        if not output or len(output.strip()) < 3:
            return False
        
        # Check length ratio
        ratio = len(output) / len(original) if original else 0
        if ratio < 0.3 or ratio > 3.0:
            return False
        
        # Check for excessive repetition
        words = output.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                return False
        
        return True
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleanup while preserving formatting"""
        if not text:
            return ""
        
        # Preserve formatting but fix common dictation issues
        # Replace multiple spaces (but not newlines) with single space
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        # Capitalize first letter if text doesn't start with formatting
        if text and not text[0].isspace():
            text = text[0].upper() + text[1:]
        
        return text
    
    async def _publish_completed(self, processed_text: str, agentic_prompt: str) -> None:
        """Publish processing completed event"""
        try:
            logger.info(f"📢 LLM SERVICE: Publishing LLMProcessingCompletedEvent with text: '{processed_text[:100]}...'")
            event = LLMProcessingCompletedEvent(
                processed_text=processed_text,
                agentic_prompt=agentic_prompt
            )
            await self.event_bus.publish(event)
            logger.info("✅ LLM SERVICE: LLMProcessingCompletedEvent published successfully")
        except Exception as e:
            logger.error(f"Event publishing error: {e}", exc_info=True)
    
    async def _publish_failed(self, error_message: str, original_text: str) -> None:
        """Publish processing failed event"""
        try:
            event = LLMProcessingFailedEvent(
                error_message=error_message,
                original_text=original_text
            )
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Event publishing error: {e}", exc_info=True)
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._model_loaded and self.llm is not None
    
    async def shutdown(self) -> None:
        """Shutdown service and properly cleanup LLM model"""
        try:
            logger.info("Shutting down LLM service...")
            self._model_loaded = False
            
            with self._processing_lock:
                if self.llm:
                    # Properly close the Llama model to free memory
                    try:
                        # Llama-cpp-python models need to be explicitly closed
                        if hasattr(self.llm, 'close'):
                            self.llm.close()
                        elif hasattr(self.llm, '__del__'):
                            # Force garbage collection of the model
                            del self.llm
                        self.llm = None
                        logger.info("LLM model properly closed and freed")
                    except Exception as model_cleanup_error:
                        logger.error(f"Error closing LLM model: {model_cleanup_error}")
                        # Force deletion even if close fails
                        self.llm = None
                
                # Clear model downloader references
                if hasattr(self, 'model_downloader'):
                    self.model_downloader = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("LLM service shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True) 