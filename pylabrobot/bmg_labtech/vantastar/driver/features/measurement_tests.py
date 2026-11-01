"""Tests for what the measurement features share: the plate geometry the reader takes from a
PyLabRobot plate, and the order of measured wells."""

import unittest

from pylabrobot.bmg_labtech.vantastar.driver.features.measurement import (
  _get_measured_wells,
  get_plate_geometry,
)
from pylabrobot.resources import cor_96_wellplate_2mL_Vb, cor_96_wellplate_360uL_Fb


class PlateTests(unittest.TestCase):
  """The plate geometry the reader takes from a PyLabRobot plate, and the order of measured
  wells."""

  def test_geometry_from_the_back_left_corner(self):
    """Well centres are measured from the plate's left and back edges: A1 lies in the back half, the
    last well 99 mm right of it and 63 mm in front of it."""
    plate = cor_96_wellplate_360uL_Fb(name="plate")
    geometry = get_plate_geometry(plate)
    self.assertEqual((geometry.columns, geometry.rows), (12, 8))
    self.assertAlmostEqual(geometry.last_x - geometry.first_x, 99.0, places=1)
    self.assertAlmostEqual(geometry.last_y - geometry.first_y, 63.0, places=1)
    self.assertLess(geometry.first_y, geometry.width / 2)

  def test_square_wells(self):
    """Round wells are sent as shape 0, square wells as shape 1."""
    self.assertEqual(get_plate_geometry(cor_96_wellplate_360uL_Fb(name="plate")).well_shape, 0)
    self.assertEqual(get_plate_geometry(cor_96_wellplate_2mL_Vb(name="plate")).well_shape, 1)

  def test_measured_wells_are_in_row_order(self):
    """Measured wells come in the reader's data order, row by row, as 0-based (row, column)."""
    plate = cor_96_wellplate_360uL_Fb(name="plate")
    wells = plate.get_items(["B2", "A3", "A1"])
    self.assertEqual(_get_measured_wells(plate, wells), [(0, 0), (0, 2), (1, 1)])


if __name__ == "__main__":
  unittest.main()
