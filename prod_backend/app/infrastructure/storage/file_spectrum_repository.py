from app.domain.repositories.spectrum_repository import SpectrumRepository
from app.domain.models.spectrum import Spectrum
from app.config.settings import Settings, get_settings
from app.shared.utils.validators import validate_spectrum
from app.config.logging import get_logger
from app.core.exceptions import FileReadException, SpectrumValidationException
from app.infrastructure.ml.data_processor import DashSpectrumProcessor
import json
import os
import uuid
import urllib3
import requests
from typing import Any, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = get_logger(__name__)

class FileSpectrumRepository(SpectrumRepository):
    """
    File-based repository for spectra. Stores spectra as JSON files in a directory.
    Uses DashSpectrumProcessor to parse files.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.processor = DashSpectrumProcessor(w0=4000, w1=9000, nw=1024)
        self.storage_dir = os.path.join(self.config.storage_dir, "spectra")
        # Spectra are not persisted to disk; leave makedirs commented out to avoid creating the dir.
        # os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, spectrum: Spectrum) -> Spectrum:
        try:
            validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
            if not spectrum.id:
                spectrum.id = str(uuid.uuid4())
            # Persisting spectra to disk is disabled; return spectrum in-memory only.
            # path = os.path.join(self.storage_dir, f"{spectrum.id}.json")
            # with open(path, "w") as f:
            #     json.dump({
            #         "id": spectrum.id,
            #         "osc_ref": spectrum.osc_ref,
            #         "file_name": spectrum.file_name,
            #         "x": spectrum.x,
            #         "y": spectrum.y,
            #         "redshift": spectrum.redshift,
            #         "meta": spectrum.meta
            #     }, f)
            # logger.debug(f"Saved spectrum {spectrum.id} to {path}")
            return spectrum
        except Exception as e:
            logger.error(f"Error saving spectrum: {e}", exc_info=True)
            raise

    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """Persistence disabled: do not read from file storage."""
        # path = os.path.join(self.storage_dir, f"{spectrum_id}.json")
        # if not os.path.exists(path):
        #     return None
        # with open(path, "r") as f:
        #     data = json.load(f)
        # return Spectrum(
        #     id=data["id"],
        #     osc_ref=data.get("osc_ref"),
        #     file_name=data.get("file_name"),
        #     x=data["x"],
        #     y=data["y"],
        #     redshift=data.get("redshift"),
        #     meta=data.get("meta", {})
        # )
        return None

    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        # Not implemented for file-based repo
        return None

    def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # Accepts UploadFile or file-like object
        filename = getattr(file, 'filename', 'unknown')
        logger.debug(f"Reading spectrum file: {filename}")

        try:
            import pandas as pd
            import io
            import numpy as np

            # Handle file reading like the old backend
            file_obj = file
            if hasattr(file, 'filename') and hasattr(file, 'file'):
                # This is a FastAPI UploadFile - get the underlying file object
                file_obj = file.file

            # Handle different file types like the old backend
            if filename.lower().endswith('.lnw'):
                return self._read_lnw_file(file_obj, filename)
            elif filename.lower().endswith(('.dat', '.txt', '.ascii', '.asci', '.flm')):
                return self._read_text_file(file_obj, filename)
            elif filename.lower().endswith('.fits'):
                return self._read_fits_file(file_obj, filename)
            elif filename.lower().endswith('.csv'):
                return self._read_csv_file(file_obj, filename)
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}", exc_info=True)
            return None

    def _read_lnw_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .lnw file with specific wavelength filtering like the old backend."""
        try:
            import re

            # Read file contents
            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                with open(file_obj, 'r') as f:
                    content = f.read()

            # Parse like the old backend
            lines = content.splitlines()
            spectrum = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Split by whitespace and filter by wavelength
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 2:
                        try:
                            wavelength = float(parts[0])
                            flux = float(parts[1])

                            # Apply wavelength filter like the old backend
                            if 4000 <= wavelength <= 9000:
                                spectrum.append((wavelength, flux))
                        except ValueError:
                            continue

            if not spectrum:
                logger.error(f"No valid spectrum data found in {filename}")
                return None

            # Sort by wavelength and separate arrays
            spectrum.sort(key=lambda x: x[0])
            wavelength = [x[0] for x in spectrum]
            flux = [x[1] for x in spectrum]

            # Create spectrum object
            spectrum_obj = Spectrum(x=list(wavelength), y=list(flux), file_name=filename)

            # Validate before saving
            try:
                validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
            except Exception as e:
                logger.error(f"Spectrum validation failed for .lnw file: {e}")
                return None

            saved_spectrum = self.save(spectrum_obj)
            return saved_spectrum

        except Exception as e:
            logger.error(f"Error reading .lnw file {filename}: {e}", exc_info=True)
            return None

    def _read_text_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .dat or .txt file: two-column wavelength/flux, filter 4000–9000 Å."""
        try:
            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                with open(file_obj, 'r', encoding='utf-8') as f:
                    content = f.read()

            lines = content.splitlines()
            spectrum_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    wavelength = float(parts[0])
                    flux = float(parts[1])
                except ValueError:
                    continue
                if 4000.0 <= wavelength <= 9000.0:
                    spectrum_data.append((wavelength, flux))

            if not spectrum_data:
                logger.error(f"No valid spectrum data in {filename} (after 4000-9000 Å filter)")
                return None

            spectrum_data.sort(key=lambda x: x[0])
            wavelength = [w for w, _ in spectrum_data]
            flux = [f for _, f in spectrum_data]
            spectrum_obj = Spectrum(x=wavelength, y=flux, file_name=filename)

            try:
                validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
            except Exception as e:
                logger.error(f"Spectrum validation failed for text file: {e}")
                return None

            return self.save(spectrum_obj)

        except Exception as e:
            logger.error(f"Error reading text file {filename}: {e}", exc_info=True)
            return None

    def _read_csv_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .csv with header row; use WAVE and FLUX columns (or first two columns)."""
        try:
            import csv
            import io

            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                with open(file_obj, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Try comma then tab
            for delimiter in (',', '\t'):
                reader = csv.reader(io.StringIO(content), delimiter=delimiter)
                rows = list(reader)
                if not rows:
                    continue
                header = [c.strip().upper() for c in rows[0]]
                data_rows = rows[1:]

                # Find WAVE and FLUX column indices (common header names)
                wave_idx = None
                flux_idx = None
                for i, col in enumerate(header):
                    if col in ('WAVE', 'WAVELENGTH', 'LAMBDA', 'WL'):
                        wave_idx = i
                    if col in ('FLUX', 'FLUX_DENSITY', 'F'):
                        flux_idx = i
                if wave_idx is None or flux_idx is None:
                    # Fallback: first two columns
                    if len(header) >= 2:
                        wave_idx, flux_idx = 0, 1
                    else:
                        continue

                spectrum_data = []
                for row in data_rows:
                    if len(row) <= max(wave_idx, flux_idx):
                        continue
                    try:
                        w = float(row[wave_idx].strip())
                        f = float(row[flux_idx].strip())
                        if 4000.0 <= w <= 9000.0:
                            spectrum_data.append((w, f))
                    except (ValueError, IndexError):
                        continue

                if not spectrum_data:
                    continue

                spectrum_data.sort(key=lambda x: x[0])
                wavelength = [x[0] for x in spectrum_data]
                flux = [x[1] for x in spectrum_data]
                spectrum_obj = Spectrum(x=wavelength, y=flux, file_name=filename)

                try:
                    validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
                except Exception as e:
                    logger.error(f"Spectrum validation failed for CSV file: {e}")
                    return None

                return self.save(spectrum_obj)

            logger.error(f"No valid spectrum data in CSV file {filename}")
            return None

        except Exception as e:
            logger.error(f"Error reading CSV file {filename}: {e}", exc_info=True)
            return None

    def _read_fits_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .fits file: table extensions or primary HDU as 1D spectrum with WCS."""
        try:
            from astropy.io import fits
            import numpy as np

            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                hdul = fits.open(file_obj)
            else:
                hdul = fits.open(file_obj)

            try:
                spectrum_data = None
                wavelength = None
                flux = None

                # Look for common spectrum extensions
                for ext in ['SPECTRUM', 'SPECTRA', 'FLUX', 'DATA']:
                    if ext in hdul:
                        spectrum_data = hdul[ext].data
                        break
                if spectrum_data is None and len(hdul) > 1:
                    spectrum_data = hdul[1].data

                # Table-like extension: wavelength + flux columns/attributes
                if spectrum_data is not None:
                    if hasattr(spectrum_data, 'wavelength') and hasattr(spectrum_data, 'flux'):
                        wavelength = np.asarray(spectrum_data.wavelength, dtype=float)
                        flux = np.asarray(spectrum_data.flux, dtype=float)
                    elif hasattr(spectrum_data, 'wave') and hasattr(spectrum_data, 'flux'):
                        wavelength = np.asarray(spectrum_data.wave, dtype=float)
                        flux = np.asarray(spectrum_data.flux, dtype=float)
                    elif getattr(spectrum_data.dtype, 'names', None) and len(spectrum_data.dtype.names) >= 2:
                        wavelength = np.asarray(spectrum_data[spectrum_data.dtype.names[0]], dtype=float)
                        flux = np.asarray(spectrum_data[spectrum_data.dtype.names[1]], dtype=float)
                    else:
                        spectrum_data = None  # fall through to primary HDU handling

                # Primary HDU as 1D image with WCS (e.g. IRAF-style spectrum)
                if spectrum_data is None and wavelength is None and len(hdul) > 0:
                    primary = hdul[0]
                    if hasattr(primary, 'data') and primary.data is not None:
                        data = np.asarray(primary.data, dtype=float).flatten()
                        if data.ndim == 1 and len(data) > 0:
                            h = primary.header
                            crval1 = h.get('CRVAL1')
                            crpix1 = h.get('CRPIX1', 1)
                            cdel1 = h.get('CDELT1')
                            if crval1 is not None and cdel1 is not None:
                                # FITS pixel indices are 1-based
                                wavelength = crval1 + (np.arange(len(data), dtype=float) + 1 - crpix1) * cdel1
                                flux = data
                            else:
                                logger.error(f"No spectrum table and no WCS (CRVAL1/CDELT1) in FITS file {filename}")
                                return None
                        else:
                            logger.error(f"No spectrum data found in FITS file {filename}")
                            return None
                    else:
                        logger.error(f"No spectrum data found in FITS file {filename}")
                        return None

                if wavelength is None or flux is None:
                    logger.error(f"No spectrum data found in FITS file {filename}")
                    return None

                # Convert to lists and apply wavelength filter
                wavelength = wavelength.tolist()
                flux = flux.tolist()
                filtered_data = [(w, f) for w, f in zip(wavelength, flux) if 4000 <= w <= 9000]

                if not filtered_data:
                    logger.error(f"No data in wavelength range 4000-9000 in FITS file {filename}")
                    return None

                wavelength = [x[0] for x in filtered_data]
                flux = [x[1] for x in filtered_data]
                spectrum_obj = Spectrum(x=wavelength, y=flux, file_name=filename)

                try:
                    validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
                except Exception as e:
                    logger.error(f"Spectrum validation failed for FITS file: {e}")
                    return None

                return self.save(spectrum_obj)

            finally:
                hdul.close()

        except Exception as e:
            logger.error(f"Error reading FITS file {filename}: {e}", exc_info=True)
            return None


class OSCSpectrumRepository(SpectrumRepository):
    """
    Repository for retrieving spectra from the Open Supernova Catalog (OSC) API.
    """

    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        # Use configurable OSC API URL, fallback to default if not set
        self.base_url = getattr(self.config, 'osc_api_url', 'https://api.astrocats.space')
        # The working backend uses the base URL directly, not with /api suffix

    def save(self, spectrum: Spectrum) -> Spectrum:
        # OSC repository doesn't save - it only retrieves
        raise NotImplementedError("OSC repository doesn't support saving")

    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        # OSC repository uses OSC references, not internal IDs
        return None

    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        """Get spectrum from OSC API."""
        try:
            # Extract the SN name from the OSC reference format
            # Input: "osc-sn2002er-0" -> Extract: "sn2002er" -> Convert to: "SN2002ER"
            if osc_ref.startswith('osc-'):
                # Remove "osc-" prefix and "-0" suffix
                sn_name = osc_ref[4:-2]  # "osc-sn2002er-0" -> "sn2002er"
            else:
                sn_name = osc_ref

            # The API expects the object name in uppercase
            obj_name = sn_name.upper()

            # Use the correct API structure: /{OBJECT_NAME}/spectra/time+data
            url = f"{self.base_url}/{obj_name}/spectra/time+data"
            logger.debug(f"OSC repository: Attempting to fetch spectrum from {url}")

            response = requests.get(url, verify=False, timeout=30)
            logger.debug(f"OSC repository: Received response status {response.status_code} for {osc_ref}")

            if response.status_code != 200:
                logger.error(f"OSC API returned status {response.status_code} for {osc_ref}")
                logger.error(f"Response content: {response.text[:500]}")  # Log first 500 chars of response
                return None

            data = response.json()
            logger.debug(f"OSC repository: Raw API response for {osc_ref}: {data}")

            # Resolve API key case-insensitively (match comparison script) so we load the same spectrum
            if obj_name not in data:
                for key in data.keys():
                    if getattr(key, "upper", lambda: key)() == obj_name:
                        obj_name = key
                        break
                else:
                    logger.error(f"OSC API response has no key for {sn_name.upper()}")
                    return None

            # Parse using the actual API response structure
            try:
                # The API returns: {"SN2002ER": {"spectra": [["52512", [["wavelength", "flux"], ...]]]}}
                # We need: data[obj_name]["spectra"][0][1] to get the spectrum data array
                spectrum_data = data[obj_name]["spectra"][0][1]

                # Convert to numpy arrays and transpose to get wave, flux
                import numpy as np
                wave, flux = np.array(spectrum_data).T.astype(float)

                logger.debug(f"OSC repository: Successfully parsed spectrum data for {osc_ref}")

            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to parse spectrum data structure for {osc_ref}: {e}")
                logger.error(f"Response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                return None

            # Fetch redshift from OSC API (same as comparison script) so web matches script results
            redshift = 0.0
            try:
                # Use same key as spectra (API may be case-sensitive)
                z_url = f"{self.base_url}/{obj_name}/redshift"
                z_response = requests.get(z_url, verify=False, timeout=10)
                if z_response.status_code == 200:
                    z_data = z_response.json()
                    if obj_name in z_data and "redshift" in z_data[obj_name]:
                        z_list = z_data[obj_name]["redshift"]
                        if z_list and len(z_list) > 0:
                            redshift = float(z_list[0]["value"])
                            logger.debug(f"OSC repository: Fetched redshift {redshift} for {osc_ref}")
            except Exception as e:
                logger.debug(f"OSC repository: Could not fetch redshift for {osc_ref}: {e}, using 0.0")

            # Extract object name for filename
            obj_name = data.get("name", obj_name)

            # Generate spectrum ID using just the SN name to avoid duplicates
            spectrum_id = f"osc_{sn_name}"

            # Create spectrum object
            spectrum = Spectrum(
                id=spectrum_id,
                x=wave.tolist(),  # Convert numpy array to list
                y=flux.tolist(),   # Convert numpy array to list
                redshift=redshift,
                osc_ref=osc_ref,
                file_name=f"{obj_name}.json",
                meta={"source": "osc", "object_name": obj_name}
            )

            logger.debug(f"OSC repository: Created spectrum object: {spectrum}")

            # Validate spectrum
            try:
                validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
                logger.debug(f"OSC repository: Spectrum validation passed")
            except Exception as e:
                logger.error(f"OSC repository: Spectrum validation failed: {e}")
                return None

            return spectrum

        except requests.RequestException as e:
            logger.error(f"OSC API request failed for {osc_ref}: {e}")
            logger.error(f"OSC API URL being used: {self.base_url}")
            logger.error(f"Full OSC reference: {osc_ref}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving spectrum from OSC for {osc_ref}: {e}", exc_info=True)
            return None

    def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # OSC repository doesn't read files
        return None
