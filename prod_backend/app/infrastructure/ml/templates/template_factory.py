"""
Factory function for creating appropriate template handlers.
Provides a centralized way to instantiate template handlers based on model type.
"""

from typing import Optional
import os
from app.infrastructure.ml.templates.dash_template_handler import DASHSpectrumTemplate
from app.infrastructure.ml.templates.transformer_template_handler import TransformerSpectrumTemplate
from app.infrastructure.ml.templates.template_interface import SpectrumTemplateInterface
from app.core.exceptions import FileNotFoundException, ModelConfigurationException

def create_spectrum_template_handler(model_type: str, template_path: Optional[str] = None) -> SpectrumTemplateInterface:
    """
    Factory function to create appropriate template handler.

    Args:
        model_type: Type of model ('dash' for DASH, 'transformer' for Transformer)
        template_path: Path to template file (required for DASH)

    Returns:
        Appropriate template handler instance

    Raises:
        FileNotFoundException: If template file doesn't exist
        ModelConfigurationException: If model type is unsupported
    """
    if model_type == 'dash':
        if not template_path:
            # Get template path from settings
            from app.config.settings import get_settings
            settings = get_settings()
            template_path = settings.template_path

        if not os.path.exists(template_path):
            raise FileNotFoundException(template_path)

        return DASHSpectrumTemplate(template_path)

    elif model_type == 'transformer':
        return TransformerSpectrumTemplate()

    else:
        raise ModelConfigurationException(f"Unsupported model type: {model_type}")
