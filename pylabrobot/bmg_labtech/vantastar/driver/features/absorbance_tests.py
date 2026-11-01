"""Absorbance tests through the device, over the driver's fake USB link with the reader's captured
replies, against the values Voyager exported for the same reads."""

import unittest
from typing import Any, Dict

from pylabrobot.bmg_labtech.vantastar.device_tests import (
  PHOTOCATALYST_WELLS,
  make_device,
  make_recorded_device,
)
from pylabrobot.bmg_labtech.vantastar.driver.responses_tests import (
  ABSORBANCE_FIRST_PAGE,
  ABSORBANCE_LAST_PAGE,
  VOYAGER_OD600,
  load_flows,
)


class AbsorbanceTests(unittest.IsolatedAsyncioTestCase):
  """Absorbance reads: an endpoint read, and recorded reads of a photocatalyst plate
  (``load_flows``)."""

  flows: Dict[str, Any]

  @classmethod
  def setUpClass(cls):
    """Load the recorded reads once for all the tests."""
    cls.flows = load_flows()

  async def test_absorbance_results(self):
    """An absorbance read at 600 nm gives the temperature and Voyager's optical densities, keyed by
    well, row by row."""
    device = make_device([ABSORBANCE_FIRST_PAGE, ABSORBANCE_LAST_PAGE])
    (result,) = await device.absorbance.measure([600])
    self.assertEqual(result.wavelength, 600)
    self.assertEqual(result.temperature, 26.0)
    self.assertEqual(list(result.data)[:13], [f"A{c}" for c in range(1, 13)] + ["B1"])
    self.assertEqual([round(v, 3) for v in result.data.values()], VOYAGER_OD600)

  async def test_absorbance_at_three_wavelengths(self):
    """An absorbance read at 300, 450 and 600 nm gives the optical densities Voyager exported for
    each wavelength."""
    device = make_recorded_device("absorbance_three_wavelengths")
    results = await device.absorbance.measure([300, 450, 600])
    voyager = self.flows["absorbance_three_wavelengths"]["voyager_optical_densities"]
    for mode, result in enumerate(results, start=1):
      self.assertEqual(
        {w: round(min(v, 3.5), 3) for w, v in result.data.items()}, voyager[str(mode)]
      )

  async def test_absorbance_kinetic(self):
    """A kinetic absorbance read gives each cycle's optical densities as Voyager exported them, the
    cycles starting 10 s apart."""
    device = make_recorded_device("absorbance_kinetic")
    plate = device.plate_holder.resource
    assert plate is not None
    cycles = await device.absorbance.measure_kinetic(
      [450], cycles=6, cycle_time=10, wells=plate.get_items(["A1", "A6", "A12"])
    )
    voyager = self.flows["absorbance_kinetic"]["voyager_optical_densities"]
    for k, (result,) in enumerate(cycles):
      self.assertEqual(
        {w: round(v, 3) for w, v in result.data.items()}, {w: voyager[w][k] for w in result.data}
      )
    starts = [result.start_timestamp for (result,) in cycles]
    self.assertEqual([(s - starts[0]).total_seconds() for s in starts], [0, 10, 20, 30, 40, 50])

  async def test_absorbance_spectrum(self):
    """An absorbance spectrum gives each well's optical densities from 220 to 1000 nm as Voyager
    exported them, the wells row by row."""
    device = make_recorded_device("absorbance_spectrum")
    plate = device.plate_holder.resource
    assert plate is not None
    result = await device.absorbance.measure_spectrum(
      220, 1000, wells=plate.get_items(PHOTOCATALYST_WELLS)
    )
    self.assertEqual(list(result.data), PHOTOCATALYST_WELLS)  # row by row
    voyager = self.flows["absorbance_spectrum"]["voyager_optical_densities"]
    for well, spectrum in result.data.items():
      self.assertEqual(list(spectrum), list(range(220, 1001)))
      # Voyager reports optical densities above 3.5 as 3.5.
      self.assertEqual([round(min(v, 3.5), 3) for v in spectrum.values()], voyager[well])


class AbsorbanceArgumentTests(unittest.IsolatedAsyncioTestCase):
  """Absorbance arguments refused before anything is sent."""

  async def test_spectrum_step_must_divide_the_range(self):
    """An absorbance spectrum is refused before anything is sent when its step does not divide the
    range or its end is not above its start."""
    device = make_device([])
    with self.assertRaises(ValueError):
      await device.absorbance.measure_spectrum(220, 1000, 7)  # 780 nm is not a multiple of 7
    with self.assertRaises(ValueError):
      await device.absorbance.measure_spectrum(600, 500)
    self.assertEqual(device.driver.io.commands, [])  # type: ignore[attr-defined]

  async def test_kinetic_needs_two_cycles(self):
    """A kinetic read of one cycle is refused."""
    device = make_device([])
    with self.assertRaises(ValueError):
      await device.absorbance.measure_kinetic([450], cycles=1, cycle_time=10)
    self.assertEqual(device.driver.io.commands, [])  # type: ignore[attr-defined]

  async def test_limits_the_reader_refuses(self):
    """Settling times outside 0.02-1.0 s, more than 500 flashes and more than 1000 kinetic cycles,
    which the reader refuses (quit codes 43 and 60), are refused before anything is sent."""
    device = make_device([])
    for settling_time in (0, 1.04):
      with self.assertRaises(ValueError):
        await device.absorbance.measure([450], settling_time=settling_time)
    with self.assertRaises(ValueError):
      await device.absorbance.measure([450], flashes=501)
    with self.assertRaises(ValueError):
      await device.absorbance.measure_kinetic([450], cycles=1001, cycle_time=10)
    self.assertEqual(device.driver.io.commands, [])  # type: ignore[attr-defined]


if __name__ == "__main__":
  unittest.main()
