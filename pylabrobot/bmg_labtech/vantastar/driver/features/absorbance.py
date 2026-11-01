"""Absorbance: endpoint, kinetic and spectrum reads through the reader's spectrometer."""

from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

from pylabrobot.lib.absorbance.results import AbsorbanceResult, AbsorbanceSpectrumResult
from pylabrobot.resources import Plate, Well

from ..responses import (
  MeasurementData,
  compute_optical_densities,
  compute_spectrum_optical_densities,
)
from .measurement import (
  Measurement,
  WellScan,
  _get_identifier,
  _get_measured_wells,
  get_plate_geometry,
)

# Absorbance wavelengths Voyager offers, in nm.
MINIMUM_ABSORBANCE_WAVELENGTH = 220
MAXIMUM_ABSORBANCE_WAVELENGTH = 1000


class Absorbance(Measurement):
  """Absorbance measurement (RUN, MeasurementMode 2).

  Reached as ``driver.absorbance``.
  """

  @staticmethod
  def _check_wavelengths(wavelengths: Sequence[int]) -> None:
    """Raise unless 1 to 8 wavelengths, each 220-1000 nm."""
    if not 1 <= len(wavelengths) <= 8:
      raise ValueError(f"1 to 8 wavelengths, got {len(wavelengths)}")
    for wavelength in wavelengths:
      if not MINIMUM_ABSORBANCE_WAVELENGTH <= wavelength <= MAXIMUM_ABSORBANCE_WAVELENGTH:
        raise ValueError(
          f"wavelength must be {MINIMUM_ABSORBANCE_WAVELENGTH}-{MAXIMUM_ABSORBANCE_WAVELENGTH} nm, "
          f"got {wavelength}"
        )

  def _get_results(
    self,
    data: MeasurementData,
    measured: List[Tuple[int, int]],
    wavelengths: Sequence[int],
    start: datetime,
    end: datetime,
  ) -> List[AbsorbanceResult]:
    """One ``AbsorbanceResult`` per wavelength of an absorbance block."""
    self._check_wells(data, measured)
    identifiers = [_get_identifier(r, c) for r, c in measured]
    return [
      AbsorbanceResult(
        data=dict(zip(identifiers, compute_optical_densities(data, i))),
        wavelength=wavelength,
        temperature=data.temperature,
        start_timestamp=start,
        end_timestamp=end,
      )
      for i, wavelength in enumerate(wavelengths)
    ]

  async def measure(
    self,
    wavelengths: Sequence[int],
    *,
    plate: Optional[Plate] = None,
    wells: Optional[Sequence[Well]] = None,
    flashes: int = 22,
    settling_time: float = 0.5,
    well_scan: Optional[WellScan] = None,
    timeout: float = 300.0,
  ) -> List[AbsorbanceResult]:
    """Measure the absorbance of the plate at each wavelength (endpoint).

    The defaults are Voyager's "precise" preset.  The optical density is computed as Voyager
    computes it, against the reader's 100% and reference values; it is not pathlength-corrected.
    Voyager reports optical densities above 3.5 as 3.5; these are the computed values, ``inf`` for a
    well that lets no measurable light through.  Averaging over an orbit or spiral reads higher
    optical densities than the centre, near the well's edge (Voyager's too).

    Args:
      wavelengths: 1 to 8 wavelengths in nm (220-1000).
      plate: the plate to measure; None uses the plate on the plate holder.
      wells: the wells to measure; None measures every well.
      flashes: flashes per well (1-500).
      settling_time: seconds to settle before reading each well (0.02-1.0, 20 ms steps).
      well_scan: average over an orbit or spiral in each well; None reads the centre.
      timeout: seconds the run may take.

    Returns:
      One ``AbsorbanceResult`` per wavelength, in the order given.

    Raises:
      RuntimeError: if no plate is on the plate holder or the run ends without its data.
      ValueError: if an argument is out of range.
    """
    self._check_wavelengths(wavelengths)
    self._check_flashes(flashes)
    self._check_settling_time(settling_time)
    plate = self._get_plate(plate)
    measured = _get_measured_wells(plate, wells)
    parameters = self._driver.build_absorbance_run(
      wavelengths,
      flashes=flashes,
      settling_time=settling_time,
      geometry=get_plate_geometry(plate),
      well_mask=self._get_well_mask(plate, wells),
      well_scan=well_scan,
    )
    start = datetime.now()
    data = await self._driver.run_measurement(parameters, timeout=timeout)
    end = datetime.now()
    return self._get_results(data, measured, wavelengths, start, end)

  async def measure_kinetic(
    self,
    wavelengths: Sequence[int],
    *,
    cycles: int,
    cycle_time: int,
    plate: Optional[Plate] = None,
    wells: Optional[Sequence[Well]] = None,
    flashes: int = 22,
    settling_time: float = 0.5,
    well_scan: Optional[WellScan] = None,
    timeout: Optional[float] = None,
  ) -> List[List[AbsorbanceResult]]:
    """Measure the absorbance of the plate at each wavelength ``cycles`` times, the reader starting
    a cycle every ``cycle_time`` seconds (plate-mode kinetic, as ``measure``).

    Args:
      wavelengths: 1 to 8 wavelengths in nm (220-1000).
      cycles: reads of the plate (2-1000).
      cycle_time: whole seconds from one cycle's start to the next (1-65535); the reader refuses a
        cycle time shorter than one cycle takes (``CommandRefusedError``, quit code 60).
      plate: the plate to measure; None uses the plate on the plate holder.
      wells: the wells to measure; None measures every well.
      flashes: flashes per well (1-500).
      settling_time: seconds to settle before reading each well (0.02-1.0, 20 ms steps).
      well_scan: average over an orbit or spiral in each well; None reads the centre.
      timeout: seconds the run may take; None allows the cycles plus 300 s.

    Returns:
      Per cycle, one ``AbsorbanceResult`` per wavelength in the order given; a cycle's
      ``start_timestamp`` is when the reader started it, its ``end_timestamp`` when the run's data
      was in.

    Raises:
      RuntimeError: if no plate is on the plate holder or the run ends without its data.
      ValueError: if an argument is out of range.
    """
    self._check_wavelengths(wavelengths)
    self._check_cycles(cycles, cycle_time)
    self._check_flashes(flashes)
    self._check_settling_time(settling_time)
    plate = self._get_plate(plate)
    measured = _get_measured_wells(plate, wells)
    parameters = self._driver.build_absorbance_run(
      wavelengths,
      flashes=flashes,
      settling_time=settling_time,
      geometry=get_plate_geometry(plate),
      well_mask=self._get_well_mask(plate, wells),
      well_scan=well_scan,
      cycles=cycles,
      cycle_time=cycle_time,
    )
    start = datetime.now()
    blocks = await self._driver.run_blocks(
      parameters, timeout=cycles * cycle_time + 300.0 if timeout is None else timeout
    )
    end = datetime.now()
    return [
      self._get_results(
        block, measured, wavelengths, start + timedelta(seconds=block.start_time), end
      )
      for block in blocks
    ]

  async def measure_spectrum(
    self,
    start_wavelength: int,
    end_wavelength: int,
    step_size: int = 1,
    *,
    plate: Optional[Plate] = None,
    wells: Optional[Sequence[Well]] = None,
    flashes: int = 22,
    settling_time: float = 0.5,
    well_scan: Optional[WellScan] = None,
    timeout: float = 600.0,
  ) -> AbsorbanceSpectrumResult:
    """Measure the absorbance spectrum of the plate: every wavelength at once, well by well (the
    reader's spectrometer).  The optical densities are computed as in ``measure``.

    Args:
      start_wavelength: the first wavelength in nm (220-1000).
      end_wavelength: the last wavelength in nm, above ``start_wavelength`` (220-1000).
      step_size: nm between wavelengths, a whole nm dividing the range.
      plate: the plate to measure; None uses the plate on the plate holder.
      wells: the wells to measure; None measures every well.
      flashes: flashes per well (1-500).
      settling_time: seconds to settle before reading each well (0.02-1.0, 20 ms steps).
      well_scan: average over an orbit or spiral in each well; None reads the centre.
      timeout: seconds the run may take.

    Returns:
      Each measured well's spectrum, keyed by well identifier.

    Raises:
      RuntimeError: if no plate is on the plate holder or the run ends without its data.
      ValueError: if an argument is out of range.
    """
    self._check_wavelengths([start_wavelength, end_wavelength])
    scanned = self._get_scan_wavelengths(start_wavelength, end_wavelength, step_size)
    self._check_flashes(flashes)
    self._check_settling_time(settling_time)
    plate = self._get_plate(plate)
    measured = _get_measured_wells(plate, wells)
    parameters = self._driver.build_absorbance_spectrum_run(
      start_wavelength,
      end_wavelength,
      step_size,
      flashes=flashes,
      settling_time=settling_time,
      geometry=get_plate_geometry(plate),
      well_mask=self._get_well_mask(plate, wells),
      well_scan=well_scan,
    )
    start = datetime.now()
    blocks = await self._driver.run_absorbance_spectrum(parameters, timeout=timeout)
    end = datetime.now()
    positions: List[Tuple[int, int]] = []
    for block in blocks:
      if block.well is None:
        raise RuntimeError("an absorbance spectrum block holds no single well")
      positions.append((block.well[1] - 1, block.well[0] - 1))
    if sorted(positions) != measured:
      raise RuntimeError(f"the reader sent wells {positions}, {measured} were asked for")
    spectra = compute_spectrum_optical_densities(blocks, len(scanned))
    data = {
      _get_identifier(*position): dict(zip(scanned, spectrum))
      for position, spectrum in sorted(zip(positions, spectra))
    }
    return AbsorbanceSpectrumResult(
      data=data,
      temperature=blocks[0].temperature,
      start_timestamp=start,
      end_timestamp=end,
    )
