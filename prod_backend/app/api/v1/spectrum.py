from fastapi import APIRouter, UploadFile, File, Form, Depends, Query
from typing import Optional, Dict, Any, List
from app.shared.schemas.spectrum import SpectrumSchema
from app.shared.schemas.classification import ClassificationSchema
from app.core.dependencies import get_sqlalchemy_spectrum_repository, get_osc_spectrum_repo, get_model_factory, get_app_settings, get_template_analysis_service, get_file_spectrum_repo, get_line_list_service, get_spectrum_processing_service
from app.domain.services.spectrum_service import SpectrumService
from app.domain.services.classification_service import ClassificationService
from app.domain.services.spectrum_processing_service import SpectrumProcessingService
from app.domain.services.template_analysis_service import TemplateAnalysisService
from app.domain.repositories.spectrum_repository import create_spectrum_template_handler
from app.domain.models.spectrum import Spectrum
from app.shared.utils.validators import ValidationError
from app.domain.services.redshift_service import RedshiftService
from app.shared.utils.helpers import prepare_log_wavelength_and_templates, get_nonzero_minmax, sanitize_for_json, construct_osc_reference
from app.config.logging import get_logger
from app.core.exceptions import (
    TemplateNotFoundException,
    LineListNotFoundException,
    ElementNotFoundException,
    ValidationException,
    SpectrumProcessingException,
    ConfigurationException
)
import json
import io
import numpy as np
import os
from app.domain.services.line_list_service import LineListService


logger = get_logger(__name__)
router = APIRouter()

@router.get("/analysis-options")
async def get_analysis_options(
    analysis_service: TemplateAnalysisService = Depends(get_template_analysis_service)
):
    """
    Get available SN types and age bins for analysis.

    Returns valid supernova types and their corresponding age bins
    that can be used for spectrum processing and classification.
    """
    try:
        logger.info("Requested analysis options")

        options = await analysis_service.get_analysis_options()

        logger.info(f"Analysis options returned {len(options['sn_types'])} SN types")

        return {
            'sn_types': options['sn_types'],
            'age_bins_by_type': options['age_bins_by_type']
        }

    except Exception as e:
        logger.error(f"Error fetching analysis options: {e}", exc_info=True)
        raise ConfigurationException(f"Failed to fetch analysis options: {str(e)}")

@router.get("/template-spectrum", response_model=SpectrumSchema)
async def get_template_spectrum(sn_type: str = Query('Ia'), age_bin: str = Query('2 to 6'), settings = Depends(get_app_settings)):
    """Get template spectrum for a given SN type and age bin."""
    try:
        from urllib.parse import unquote_plus
        sn_type_decoded = unquote_plus(sn_type)
        age_bin_decoded = unquote_plus(age_bin)
        logger.info(f"Requested sn_type: {repr(sn_type_decoded)}")
        logger.info(f"Requested age_bin: {repr(age_bin_decoded)}")

        npz_path = settings.template_path
        logger.info(f"Looking for template file at: {npz_path}")
        logger.info(f"File exists: {os.path.exists(npz_path)}")

        if not os.path.exists(npz_path):
            logger.error(f"Template data file not found: {npz_path}")
            raise TemplateNotFoundException(sn_type_decoded, age_bin_decoded)

        data = np.load(npz_path, allow_pickle=True)
        logger.info(f"Loaded npz file, keys: {list(data.keys())}")

        snTemplates_raw = data['snTemplates'].item() if 'snTemplates' in data else data['arr_0'].item()
        snTemplates = {str(k): v for k, v in snTemplates_raw.items()}

        logger.info(f"Available SN types: {list(snTemplates.keys())}")
        logger.info(f"Looking for SN type: {sn_type_decoded}")

        if sn_type_decoded not in snTemplates:
            logger.error(f"SN type '{sn_type_decoded}' not found in available types: {list(snTemplates.keys())}")
            raise TemplateNotFoundException(sn_type_decoded)

        if not isinstance(snTemplates[sn_type_decoded], dict):
            snTemplates[sn_type_decoded] = dict(snTemplates[sn_type_decoded])

        logger.info(f"Available age bins for {sn_type_decoded}: {list(snTemplates[sn_type_decoded].keys())}")
        logger.info(f"Looking for age bin: {age_bin_decoded}")

        if age_bin_decoded not in snTemplates[sn_type_decoded].keys():
            logger.error(f"Age bin '{age_bin_decoded}' not found for SN type '{sn_type_decoded}'.")
            raise TemplateNotFoundException(sn_type_decoded, age_bin_decoded)

        snInfo = snTemplates[sn_type_decoded][age_bin_decoded].get('snInfo', None)
        if not isinstance(snInfo, np.ndarray) or snInfo.shape[0] == 0:
            logger.error(f"No template spectrum available for SN type '{sn_type_decoded}' and age bin '{age_bin_decoded}'.")
            raise TemplateNotFoundException(sn_type_decoded, age_bin_decoded)

        logger.info(f"snInfo shape: {snInfo.shape}, type: {type(snInfo)}")
        logger.info(f"Loading only the first template (index 0) out of {snInfo.shape[0]} available templates")

        # Only load first, placeholder for now
        template = snInfo[0]
        logger.info(f"template shape: {template.shape if hasattr(template, 'shape') else 'no shape'}, type: {type(template)}")

        # template[0] is wavelength, template[1] is flux
        wave = template[0]
        flux = template[1]

        logger.info(f"wave shape: {wave.shape if hasattr(wave, 'shape') else 'no shape'}, type: {type(wave)}")
        logger.info(f"flux shape: {flux.shape if hasattr(flux, 'shape') else 'no shape'}, type: {type(flux)}")

        logger.info(f"Template spectrum loaded for {sn_type_decoded} / {age_bin_decoded} (first template only)")
        return SpectrumSchema(x=wave.tolist(), y=flux.tolist())

    except Exception as e:
        logger.error(f"Error loading template spectrum: {e}", exc_info=True)
        raise ConfigurationException(f"Error loading template spectrum: {str(e)}")

@router.get("/template-statistics")
async def get_template_statistics(
    analysis_service: TemplateAnalysisService = Depends(get_template_analysis_service)
):
    """
    Get statistics about available templates.

    Returns information about template validation rates and counts.
    """
    try:
        logger.info("Requested template statistics")

        # Get statistics using injected service
        stats = await analysis_service.get_template_statistics()

        logger.info(f"Template statistics: {stats['valid_sn_types']}/{stats['total_sn_types']} valid SN types")

        return stats

    except Exception as e:
        logger.error(f"Error fetching template statistics: {e}", exc_info=True)
        raise ConfigurationException(f"Failed to fetch template statistics: {str(e)}")

@router.post("/process")
async def process_spectrum(
    params: str = Form('{}'),
    file: Optional[UploadFile] = File(None),
    model_id: Optional[str] = Form(None),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    file_repo = Depends(get_file_spectrum_repo),
    osc_repo = Depends(get_osc_spectrum_repo),
    model_factory = Depends(get_model_factory),
    settings = Depends(get_app_settings),
    processing_service: SpectrumProcessingService = Depends(get_spectrum_processing_service)
):
    """
    Process spectrum for classification with support for:
    - File uploads (.dat, .lnw, .txt)
    - OSC references
    - User-uploaded models
    - Dash and Transformer models
    - Parameter customization (smoothing, redshift, etc.)
    """
    try:
        logger.info("/process endpoint called")
        logger.info(f"Received params: {params}")
        logger.info(f"Received file: {file.filename if file else 'None'}")
        logger.info(f"Received model_id: {model_id}")

        # Parse parameters
        parsed_params = json.loads(params) if params else {}
        logger.info(f"Parsed params: {parsed_params}")

        # Determine model type
        if model_id:
            model_type = "user_uploaded"
            logger.info(f"Using user-uploaded model: {model_id}")
        else:
            model_type = parsed_params.get('modelType', 'dash')
            if model_type not in ['dash', 'transformer']:
                model_type = 'dash'
            logger.info(f"Using model type: {model_type}")

        # Initialize services
        spectrum_service = SpectrumService(file_repo, osc_repo)
        classification_service = ClassificationService(model_factory)

        # Get spectrum data, construct OSC ref if needed
        osc_ref = parsed_params.get('oscRef')
        if osc_ref:
            osc_ref = construct_osc_reference(osc_ref)
            logger.info(f"Constructed OSC reference: {osc_ref}")

        spectrum = await spectrum_service.get_spectrum_data(
            file=file,
            osc_ref=osc_ref
        )

        # Process spectrum with parameters
        processed_spectrum = await processing_service.process_spectrum_with_params(
            spectrum=spectrum,
            params=parsed_params
        )

        # Classify spectrum
        result = await classification_service.classify_spectrum(
            spectrum=processed_spectrum,
            model_type=model_type,
            user_model_id=model_id,
            params=parsed_params
        )

        # Convert to dict and sanitize numpy types
        result_dict = result.__dict__.copy()
        result_dict['results'] = sanitize_for_json(result_dict['results'])
        result_dict['meta'] = sanitize_for_json(result_dict.get('meta', {}))

        # Return spectrum data and classification results as expected by frontend
        return {
            "spectrum": {
                "x": processed_spectrum.x,
                "y": processed_spectrum.y,
                "redshift": processed_spectrum.redshift
            },
            "classification": result_dict['results'],
            "model_type": result_dict['model_type']
        }

    except ValidationError as e:
        logger.warning(f"Validation error in /process: {e}")
        raise ValidationException(str(e))
    except Exception as e:
        logger.error(f"Error in /process: {e}", exc_info=True)
        raise SpectrumProcessingException(f"Processing error: {str(e)}")

@router.post("/estimate-redshift")
async def estimate_redshift(
    file: Optional[UploadFile] = File(None),
    sn_type: str = Form(...),
    age_bin: str = Form(...),
    redshift_service: RedshiftService = Depends(RedshiftService),
):
    """
    Estimate redshift from a spectrum using DASH (CNN) templates.

    IMPORTANT: This endpoint is only available for DASH (CNN) models as they are
    the only models that have the required spectral templates for redshift estimation.

    Supports file upload method:
    - file: Spectrum file (required)
    - sn_type: Supernova type (e.g., 'Ia', 'Ib', 'II')
    - age_bin: Age bin (e.g., '2 to 6')

    Returns:
        Dictionary with estimated redshift and error
    """
    try:
        logger.info(f"Redshift estimation requested for {sn_type} {age_bin} (DASH only)")

        # Handle file upload
        if file:
            logger.info(f"Processing uploaded file: {file.filename}")
            contents = await file.read()
            file_stream = io.StringIO(contents.decode("utf-8"))
            lines = [line.strip() for line in file_stream if line.strip() and not line.startswith("#")]
            x, y = [], []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))
                    except Exception:
                        continue
        else:
            raise ValidationException("File upload is required for redshift estimation.")

        if not x or not y:
            raise ValidationException(
                "No valid spectrum data found in file"
            )

        # Estimate redshift using DASH templates only
        result = await redshift_service.estimate_redshift_from_spectrum(
            x, y, sn_type, age_bin, model_type="dash"
        )

        logger.info(f"Redshift estimation completed: {result.get('estimated_redshift')}")
        return sanitize_for_json(result)

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error in redshift estimation: {e}", exc_info=True)
        raise SpectrumProcessingException(
            f"Redshift estimation failed: {str(e)}"
        )

@router.get("/line-list")
async def get_line_list(
    line_list_service: LineListService = Depends(get_line_list_service)
):
    """
    Return the element/ion line list for plotting vertical lines.
    """
    try:
        logger.info("Line list endpoint called")
        line_dict = line_list_service.get_line_list()
        logger.info(f"Successfully returned line list with {len(line_dict)} elements/ions")
        return line_dict
    except LineListNotFoundException as e:
        logger.error(f"Line list file not found: {e}")
        raise LineListNotFoundException()
    except Exception as e:
        logger.error(f"Unexpected error in line list endpoint: {e}", exc_info=True)
        raise ConfigurationException(f"Internal server error while loading line list: {str(e)}")

@router.get("/line-list/elements")
async def get_available_elements(
    line_list_service: LineListService = Depends(get_line_list_service)
):
    """
    Get the list of all available elements/ions in the line list.
    """
    try:
        elements = line_list_service.get_available_elements()
        return {"elements": elements}
    except Exception as e:
        logger.error(f"Error getting available elements: {e}", exc_info=True)
        raise ConfigurationException(f"Internal server error while getting available elements: {str(e)}")

@router.get("/line-list/element/{element}")
async def get_element_wavelengths(
    element: str,
    line_list_service: LineListService = Depends(get_line_list_service)
):
    """
    Get wavelengths for a specific element/ion.
    """
    try:
        wavelengths = line_list_service.get_element_wavelengths(element)
        return {"element": element, "wavelengths": wavelengths, "count": len(wavelengths)}
    except ElementNotFoundException as e:
        raise ElementNotFoundException(element)
    except Exception as e:
        logger.error(f"Error getting wavelengths for element {element}: {e}", exc_info=True)
        raise ConfigurationException(f"Internal server error while getting wavelengths for element {element}: {str(e)}")

@router.get("/line-list/filter")
async def filter_line_list(
    min_wavelength: float,
    max_wavelength: float,
    line_list_service: LineListService = Depends(get_line_list_service)
):
    """
    Filter the line list to only include wavelengths within a specified range.
    """
    try:
        if min_wavelength < 0 or max_wavelength < 0:
            raise ValidationException("Wavelengths must be positive values")
        if min_wavelength > max_wavelength:
            raise ValidationException("Minimum wavelength must be less than or equal to maximum wavelength")
        filtered_dict = line_list_service.filter_wavelengths_by_range(min_wavelength, max_wavelength)
        return {
            "min_wavelength": min_wavelength,
            "max_wavelength": max_wavelength,
            "elements": filtered_dict,
            "count": len(filtered_dict)
        }
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error filtering line list: {e}", exc_info=True)
        raise ConfigurationException(f"Internal server error while filtering line list: {str(e)}")
