"""
Clean Sound Recognizer Configuration
Unified configuration for ESC-50 based sound recognition system.
"""
from pydantic import BaseModel, Field
from typing import Dict

class SoundRecognizerConfig(BaseModel):
    """
    Clean sound recognizer configuration using only ESC-50 for non-target sounds.
    """
    
    # === AUDIO PROCESSING ===
    target_sample_rate: int = Field(
        16000, 
        description="Target sample rate for YAMNet (do not change)"
    )
    energy_threshold: float = Field(
        0.001, 
        description="Minimum audio energy for processing"
    )
    
    # === RECOGNITION PARAMETERS ===
    confidence_threshold: float = Field(
        0.15, 
        description="Minimum similarity for recognition (optimized for enhanced features)"
    )
    k_neighbors: int = Field(
        7, 
        description="Number of neighbors for k-NN voting (increased for better discrimination)"
    )
    vote_threshold: float = Field(
        0.35, 
        description="Minimum vote alignment percentage (optimized for enhanced voting)"
    )
    
    # === TRAINING ===
    default_samples_per_sound: int = Field(
        12, 
        description="Default training samples per sound (increased for better discrimination)"
    )
    sample_duration_sec: float = Field(
        2.0, 
        description="Duration of training samples in seconds"
    )
    
    # === ESC-50 NEGATIVE EXAMPLES ===
    max_esc50_samples_per_category: int = Field(
        15, 
        description="Max samples per ESC-50 category"
    )
    max_total_esc50_samples: int = Field(
        40, 
        description="Maximum total ESC-50 samples (2:1 negative:positive ratio)"
    )
    
    # === ESC-50 CATEGORIES ===
    esc50_categories: Dict[str, str] = Field(
        default_factory=lambda: {
            "keyboard_typing": "keyboard_typing",
            "mouse_click": "mouse_click",
            "wind": "wind",
            "breathing": "breathing",
            "coughing": "coughing",
            "brushing_teeth": "brushing_teeth",
            "drinking_sipping": "drinking_sipping"
        },
        description="ESC-50 categories used as negative examples"
    )
