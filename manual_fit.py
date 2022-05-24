import multiprocessing
from pathlib import Path
import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from diskchef.chemistry import SciKitChemistry
from diskchef.lamda.line import Line
from diskchef.physics.williams_best import WilliamsBest100au
from diskchef.uv import UVFits

from dc_fit import ModelFit, uvs

logging.basicConfig(level=logging.INFO, force=True)

ENV_VARS = dict(
    PYTHONUNBUFFERED=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1
)

for key, value in ENV_VARS.items():
    os.environ[key] = str(value)
 
# IMAGER_EXEC = "bash -lc imager -nw"
# IMAGER_EXEC = "imager.exe -nw"
IMAGER_EXEC = "imager -nw"

def model_setup(tapering_radius=150,
                inner_radius=5,
                log_gas_mass=-2,
                temperature_slope=0.55,
                atmosphere_temperature_100au=40,
                midplane_temperature_100au=20,
                inclination_deg=35.18,
                pa_deg=79.19,
                velocity=0,
                ):
    temp_dir = Path(
        f'fit/{tapering_radius}_{inner_radius}_{log_gas_mass}_{temperature_slope}_{atmosphere_temperature_100au}_{midplane_temperature_100au}_{inclination_deg}_{pa_deg}_{velocity}')
    lines = [
        Line(name='CO J=2-1', transition=2, molecule='CO'),
        Line(name='13CO J=2-1', transition=2, molecule='13CO'),
        # Line(name='C18O J=2-1', transition=2, molecule='C18O'),
        Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
        # Line(name='DCO+ J=3-2', transition=3, molecule='DCO+'),
        # Line(name='H13CO+ J=3-2', transition=3, molecule='H13CO+'),
    ]

    model = ModelFit(
        disk="DN Tau",
        directory=temp_dir,
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
            star_mass=0.52 * u.Msun,
        ),
        chemical_params=dict(
            model="co_hco+_3params.pkl"
        ),
        inclination=inclination_deg * u.deg,
        position_angle=pa_deg * u.deg,
        distance=128.22 * u.pc,
        velocity=velocity * u.km / u.s,
        npix=100,
        channels=35,
        line_window_width=7.0 * u.km / u.s,
    )
    model.run_chemistry()
    model.run_line_radiative_transfer()
    model.plot(filename=temp_dir / "model.png", species1="CO", species2="HCO+")

    return model, temp_dir


def model_chi2(dra=0, ddec=0, **kwargs):
    model, directory = model_setup(**kwargs)
    chi2 = model.chi2(uvs, dRA=dra, dDec=ddec)
    with open("chi2.txt", "a") as f:
        f.write(f"{dra} {ddec} {kwargs} {chi2}\n")
    
    for line, uv in uvs.items():
        path_model_fits = directory / f'radmc_gas/{line}_image.fits'
        path_obs_uvfits = Path(uv.path)
        path_model_to_obs = directory / f'{line}_model.uvfits'
        path_residuals = directory / f'{line}_residuals.uvfits'
        path_model_to_obs.parent.mkdir(parents=True, exist_ok=True)
        UVFits.write_visibilities_to_uvfits(path_model_fits, path_obs_uvfits, path_model_to_obs)
        UVFits.write_visibilities_to_uvfits(path_model_fits, path_obs_uvfits, path_residuals, residual=True)
        procs = UVFits.run_imaging(path_model_to_obs, f'{line}_model', imager_executable=IMAGER_EXEC,
                                   script_filename=path_model_to_obs.parent / f'{line}_model.imager', device="png")
        procs = UVFits.run_imaging(path_residuals, f'{line}_residuals', imager_executable=IMAGER_EXEC,
                                   script_filename=path_model_to_obs.parent / f'{line}_residuals.imager', device="png")
        print(procs.stderr)
        print('++++++++')
        print(procs.stdout)
        print('\n\n')
    return chi2


def model_chi2_one_param(x):
    return model_chi2(tapering_radius=160,
                      inner_radius=10,
                      log_gas_mass=-3.4,
                      temperature_slope=0.5,
                      atmosphere_temperature_100au=30,
                      midplane_temperature_100au=30,
                      inclination_deg=35.18,
                      pa_deg=79.19,
                      dra=0,
                      ddec=0,
                      velocity=x
                      )


if __name__ == '__main__':
    param = np.linspace(100, 300, 4)
    param = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    with multiprocessing.Pool(processes=8) as p:
        chi2 = p.map(model_chi2_one_param, param)
    output = Path('plots')
    output.mkdir(parents=True, exist_ok=True)
    chi2 = np.array(chi2)
    plt.scatter(param, chi2)
    plt.savefig(output / 'chi_atmosphere_t.png')
    # model, _ = model_setup()
