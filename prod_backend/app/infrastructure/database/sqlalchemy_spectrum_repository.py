from typing import Optional, Any
from sqlalchemy.orm import Session
from app.domain.models.spectrum import Spectrum
from app.domain.repositories.spectrum_repository import SpectrumRepository
from app.infrastructure.database.models import SpectrumDB
from app.config.logging import get_logger
from app.shared.utils.validators import validate_spectrum
from app.core.exceptions import SpectrumValidationException

logger = get_logger(__name__)

class SQLAlchemySpectrumRepository(SpectrumRepository):
    """SQLAlchemy-based repository for spectrum data."""

    def __init__(self, db: Session):
        self.db = db

    def save(self, spectrum: Spectrum) -> Spectrum:
        """Persistence disabled: return spectrum without writing to DB."""
        # try:
        #     validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
        #     existing = self.db.query(SpectrumDB).filter(SpectrumDB.id == spectrum.id).first()
        #     if existing:
        #         existing.osc_ref = spectrum.osc_ref
        #         existing.file_name = spectrum.file_name
        #         existing.x = spectrum.x
        #         existing.y = spectrum.y
        #         existing.redshift = spectrum.redshift
        #         existing.meta = spectrum.meta
        #         self.db.commit()
        #         logger.info(f"Updated existing spectrum {spectrum.id} in database")
        #         return spectrum
        #     else:
        #         db_spectrum = SpectrumDB(
        #             id=spectrum.id,
        #             osc_ref=spectrum.osc_ref,
        #             file_name=spectrum.file_name,
        #             x=spectrum.x,
        #             y=spectrum.y,
        #             redshift=spectrum.redshift,
        #             meta=spectrum.meta
        #         )
        #         self.db.add(db_spectrum)
        #         self.db.commit()
        #         self.db.refresh(db_spectrum)
        #         logger.info(f"Saved spectrum {spectrum.id} to database")
        #         return spectrum
        # except Exception as e:
        #     logger.error(f"Error saving spectrum to database: {e}", exc_info=True)
        #     self.db.rollback()
        #     raise
        return spectrum

    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """Persistence disabled: do not read from DB."""
        # db_spectrum = self.db.query(SpectrumDB).filter(SpectrumDB.id == spectrum_id).first()
        # return self._to_domain(db_spectrum) if db_spectrum else None
        return None

    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        """Persistence disabled: do not read from DB."""
        # db_spectrum = self.db.query(SpectrumDB).filter(SpectrumDB.osc_ref == osc_ref).first()
        # return self._to_domain(db_spectrum) if db_spectrum else None
        return None

    def get_from_file(self, file: Any) -> Optional[Spectrum]:
        """
        SQLAlchemy repository does not handle file reading.
        File reading should be handled by FileSpectrumRepository.
        This method is here only for interface compliance.
        """
        raise NotImplementedError(
            "SQLAlchemy repository should not handle file reading. "
            "Use FileSpectrumRepository for file operations."
        )

    def _to_domain(self, db_spectrum: SpectrumDB) -> Spectrum:
        """Convert database model to domain model."""
        return Spectrum(
            id=db_spectrum.id,
            osc_ref=db_spectrum.osc_ref,
            file_name=db_spectrum.file_name,
            x=db_spectrum.x,
            y=db_spectrum.y,
            redshift=db_spectrum.redshift,
            meta=db_spectrum.meta
        )
