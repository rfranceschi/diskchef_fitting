from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)

from astropy import units as u

from diskchef import Line
from diskchef.physics import WilliamsBest100au
from diskchef.chemistry import SciKitChemistry

from dc_fit import ModelFit


def main():
    # tapering_radius, inner_radius, log_gas_mass, temperature_slope, atmosphere_temperature_100au, midplane_temperature_100au
    params = [150, 5, -2, 0.55, 40, 20]

    temp_dir = Path("Reference")
    tapering_radius, inner_radius, log_gas_mass, \
    temperature_slope, atmosphere_temperature_100au, midplane_temperature_100au = params
    lines = [
        Line(name='CO J=2-1', transition=2, molecule='CO'),
        Line(name='13CO J=2-1', transition=2, molecule='13CO'),
        Line(name='C18O J=2-1', transition=2, molecule='C18O'),
        Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
        Line(name='DCO+ J=3-2', transition=3, molecule='DCO+'),
        Line(name='H13CO+ J=3-2', transition=3, molecule='H13CO+'),
    ]

    model = ModelFit(
        disk="DN Tau",
        directory=temp_dir.name,
        line_list=lines,
        physics_class=WilliamsBest100au,
        chemistry_class=SciKitChemistry,
        physics_params=dict(
            r_min=1 * u.au,
            r_max=500 * u.au,
            tapering_radius=tapering_radius * u.au,
            gas_mass=10 ** log_gas_mass * u.Msun,
            inner_radius=inner_radius * u.au,
            temperature_slope=temperature_slope,
            midplane_temperature_100au=midplane_temperature_100au * u.K,
            atmosphere_temperature_100au=atmosphere_temperature_100au * u.K,
            star_mass=0.52 * u.Msun
        ),
        chemical_params=dict(
            model="co_hco+_3params.pkl"
        ),
        inclination=35.18 * u.deg,
        position_angle=79.19 * u.deg,
        distance=128.22 * u.pc,
        npix=100,
        channels=35,
        line_window_width=6.0 * u.km / u.s,
    )
    model.run_chemistry()
    model.run_line_radiative_transfer()
    model.plot(filename="Reference/model.png", species1="CO", species2="HCO+")


if __name__ == "__main__":

    main()
