"""
Transformer-specific template handler implementation.
For now, returns empty templates as Transformer doesn't use traditional templates.
"""

from typing import Dict, Any, Tuple
import numpy as np
from app.infrastructure.ml.templates.template_interface import SpectrumTemplateInterface
from app.config.logging import get_logger

logger = get_logger(__name__)

class TransformerSpectrumTemplate(SpectrumTemplateInterface):
    """
    Transformer-specific template handler.
    For now, returns empty templates as Transformer doesn't use traditional templates.
    """

    def __init__(self):
        logger.info("TransformerSpectrumTemplate initialized")

    def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get template spectrum for Transformer model (not supported)."""
        raise NotImplementedError("Transformer model doesn't use traditional templates")

    def get_all_templates(self) -> Dict[str, Any]:
        """Get all Transformer templates (empty for now)."""
        return {}

    def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """Validate Transformer template (always False)."""
        return False
