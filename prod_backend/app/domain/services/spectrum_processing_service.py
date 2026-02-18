from typing import Dict, Any, Optional, Tuple
from app.domain.models.spectrum import Spectrum
from app.infrastructure.ml.data_processor import DashSpectrumProcessor, TransformerSpectrumProcessor
from app.shared.utils.helpers import interpolate_to_1024, normalise_spectrum
from app.config.settings import Settings, get_settings
from app.config.logging import get_logger
from app.core.exceptions import SpectrumProcessingException
import numpy as np

logger = get_logger(__name__)

class SpectrumProcessingService:
    """
    Domain service for processing spectra with custom parameters.
    Handles smoothing, redshift application, wavelength filtering, and preprocessing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.dash_processor = DashSpectrumProcessor(
            w0=self.settings.w0,
            w1=self.settings.w1,
            nw=self.settings.nw
        )
        self.transformer_processor = TransformerSpectrumProcessor(
            target_length=self.settings.nw
        )
        logger.debug("SpectrumProcessingService initialized with settings")

    async def process_spectrum_with_params(
        self,
        spectrum: Spectrum,
        params: Dict[str, Any]
    ) -> Spectrum:
        """
        Process spectrum with custom parameters.

        Args:
            spectrum: Input spectrum to process
            params: Processing parameters including:
                - smoothing: Smoothing factor (0 = no smoothing)
                - knownZ: Whether redshift is known
                - zValue: Known redshift value
                - minWave: Minimum wavelength filter
                - maxWave: Maximum wavelength filter
                - calculateRlap: Whether to calculate RLAP

        Returns:
            Processed spectrum

        Raises:
            SpectrumProcessingException: If spectrum processing fails
        """
        try:
            # Extract parameters
            smoothing = params.get('smoothing', 0)
            known_z = params.get('knownZ', False)
            z_value = params.get('zValue')
            min_wave = params.get('minWave')
            max_wave = params.get('maxWave')
            calculate_rlap = params.get('calculateRlap', False)

            # Convert spectrum data to numpy arrays
            x = np.array(spectrum.x)
            y = np.array(spectrum.y)

            logger.debug(f"Processing spectrum with {len(x)} points")
            logger.debug(f"Parameters: smoothing={smoothing}, z_value={z_value}, min_wave={min_wave}, max_wave={max_wave}")

            # Apply wavelength filtering
            if min_wave is not None or max_wave is not None:
                x, y = self._apply_wavelength_filter(x, y, min_wave, max_wave)
                logger.debug(f"Applied wavelength filter: {len(y)} points remaining")

            # Apply smoothing
            if smoothing > 0:
                y = self._apply_smoothing(x, y, smoothing)
                logger.debug(f"Applied smoothing with factor {smoothing}")

            # Normalize spectrum (matching old backend behavior)
            y = normalise_spectrum(y)
            logger.debug("Applied spectrum normalization")

            # Apply redshift from request only for non-OSC spectra (uploaded files).
            # For OSC spectra we keep the redshift fetched from the OSC API so web matches script.
            is_osc = bool(getattr(spectrum, "osc_ref", None) or (getattr(spectrum, "meta", None) or {}).get("source") == "osc")
            if z_value is not None and not is_osc:
                spectrum.redshift = float(z_value)
                logger.debug(f"Applied redshift: {z_value}")
            elif is_osc and z_value is not None:
                logger.debug(f"Keeping OSC redshift {spectrum.redshift} (ignoring request zValue={z_value})")

            # Update spectrum with processed data
            spectrum.x = x.tolist()
            spectrum.y = y.tolist()

            # Add processing metadata
            if not hasattr(spectrum, 'meta'):
                spectrum.meta = {}
            spectrum.meta.update({
                'processing_params': {
                    'smoothing': smoothing,
                    'known_z': known_z,
                    'z_value': z_value,
                    'min_wave': min_wave,
                    'max_wave': max_wave,
                    'calculate_rlap': calculate_rlap
                }
            })

            logger.debug("Spectrum processing completed successfully")
            return spectrum

        except Exception as e:
            logger.error(f"Error processing spectrum with parameters: {e}")
            raise SpectrumProcessingException(f"Spectrum processing failed: {str(e)}")

    def _apply_wavelength_filter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        min_wave: Optional[float],
        max_wave: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply wavelength filtering by zeroing flux outside the range (preserve length).
        Matches script/data_processor behavior so classification matches compare_tf_pytorch_training_set.
        """
        if min_wave is None and max_wave is None:
            return x, y

        y_out = np.copy(y)
        if min_wave is not None and np.isfinite(min_wave):
            min_idx = np.clip(np.abs(x - min_wave).argmin(), 0, len(y_out) - 1)
            y_out[:min_idx] = 0.0
        if max_wave is not None and np.isfinite(max_wave):
            max_idx = np.clip(np.abs(x - max_wave).argmin(), 0, len(y_out) - 1)
            y_out[max_idx:] = 0.0

        logger.debug(f"Wavelength filtering: zeroed flux outside range (length unchanged: {len(x)})")
        return x, y_out

    def _apply_smoothing(
        self,
        x: np.ndarray,
        y: np.ndarray,
        smoothing_factor: int
    ) -> np.ndarray:
        """
        Apply smoothing to spectrum data.

        Args:
            x: Wavelength array
            y: Flux array
            smoothing_factor: Smoothing factor (higher = more smoothing)

        Returns:
            Smoothed flux array
        """
        if smoothing_factor <= 0:
            return y

        try:
            from scipy.signal import savgol_filter

            # Calculate window length based on smoothing factor
            window_length = min(smoothing_factor * 2 + 1, len(y))
            if window_length < 3:
                window_length = 3

            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1

            # Apply Savitzky-Golay filter
            y_smoothed = savgol_filter(y, window_length, 3)

            return y_smoothed

        except ImportError:
            logger.warning("scipy not available, skipping smoothing")
            return y
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original data")
            return y

    def prepare_for_model(
        self,
        spectrum: Spectrum,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """
        Prepare spectrum data for model input.

        Args:
            spectrum: Spectrum to prepare
            model_type: Type of model ('dash' for DASH, 'transformer' for Transformer, 'user_uploaded')

        Returns:
            Dictionary with prepared data
        """
        try:
            x = np.array(spectrum.x)
            y = np.array(spectrum.y)
            z = getattr(spectrum, 'redshift', 0.0) or 0.0

            if model_type == 'dash':
                # Use DASH processor
                processed_y, min_idx, max_idx, processed_z = self.dash_processor.process(
                    wave=x,
                    flux=y,
                    z=z,
                    smooth=0,  # Smoothing already applied
                    min_wave=None,  # Filtering already applied
                    max_wave=None
                )
                return {
                    'x': x,
                    'y': processed_y,
                    'redshift': processed_z,
                    'min_idx': min_idx,
                    'max_idx': max_idx
                }

            elif model_type == 'transformer':
                # Use Transformer processor
                processed_x, processed_y, processed_z = self.transformer_processor.process(x, y, z)
                return {
                    'x': processed_x,
                    'y': processed_y,
                    'redshift': processed_z
                }

            else:
                # For user models, return as-is (they handle their own preprocessing)
                return {
                    'x': x,
                    'y': y,
                    'redshift': z
                }

        except Exception as e:
            logger.error(f"Error preparing spectrum for model {model_type}: {e}")
            raise ValueError(f"Model preparation failed: {str(e)}")
