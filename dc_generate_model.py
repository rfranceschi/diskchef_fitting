from pathlib import Path
import logging

from astropy import units as u

import diskchef
diskchef.logging_basic_config()

from diskchef import Line
from diskchef import WilliamsBest100au
from diskchef import SciKitChemistry



from dc_fit import ModelFit, model_in_directory


def main():
    # tapering_radius, inner_radius, log_gas_mass, temperature_slope, atmosphere_temperature_100au, midplane_temperature_100au, velocity
    params = [150, 5, -2, 0.55, 40, 20, 0.4]

    root = Path("Reference")

    lines = [
        Line(name='CO J=2-1', transition=2, molecule='CO'),
        Line(name='13CO J=2-1', transition=2, molecule='13CO'),
        Line(name='C18O J=2-1', transition=2, molecule='C18O'),
        Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
        Line(name='DCO+ J=3-2', transition=3, molecule='DCO+'),
        Line(name='H13CO+ J=3-2', transition=3, molecule='H13CO+'),
    ]

    model = model_in_directory(params, lines=lines, root=root)
    model.plot(filename=root / "model.png", species1="CO", species2="HCO+")

if __name__ == "__main__":
    main()
