import multiprocessing
from pathlib import Path
import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import diskchef

diskchef.logging_basic_config()
from diskchef import SciKitChemistry, Line, WilliamsBest100au, UVFits

from dc_fit import ModelFit, uvs, model_in_directory, lines

# IMAGER_EXEC = "bash -lc imager"
# IMAGER_EXEC = "imager.exe"
IMAGER_EXEC = "imager"


def model_chi2(
        tapering_radius,
        inner_radius,
        log_gas_mass,
        temperature_slope,
        atmosphere_temperature_100au,
        midplane_temperature_100au,
        inclination_deg,
        pa_deg,
        velocity,
        dra=0, ddec=0
):
    params = (tapering_radius, inner_radius, log_gas_mass, temperature_slope,
              atmosphere_temperature_100au, midplane_temperature_100au, velocity)
    root = Path(
        f'fit/{tapering_radius}_{inner_radius}_{log_gas_mass}_{temperature_slope}'
        f'_{atmosphere_temperature_100au}_{midplane_temperature_100au}_{inclination_deg}_{pa_deg}_{velocity}')

    model = model_in_directory(params=params, lines=lines, root=root)
    model.plot(filename=root / "model.png", species1="CO", species2="HCO+")

    chi2 = model.chi2(uvs, dRA=dra, dDec=ddec)
    with open("chi2.txt", "a") as f:
        f.write(f"{dra} {ddec} {params} {chi2}\n")

    for line, uv in uvs.items():
        path_model_fits = root / f'radmc_gas/{line}_image.fits'
        path_obs_uvfits = Path(uv.path)
        path_model_to_obs = root / f'{line}_model.uvfits'
        path_residuals = root / f'{line}_residuals.uvfits'
        path_model_to_obs.parent.mkdir(parents=True, exist_ok=True)
        UVFits.write_visibilities_to_uvfits(path_model_fits, path_obs_uvfits, path_model_to_obs)
        UVFits.write_visibilities_to_uvfits(path_model_fits, path_obs_uvfits, path_residuals, residual=True)
        procs = UVFits.run_imaging(path_model_to_obs, f'{line}_model', imager_executable=IMAGER_EXEC,
                                   script_filename=path_model_to_obs.parent / f'{line}_model.imager', device="png")
        print(procs.stderr)
        print('++++++++')
        print(procs.stdout)
        print('++++++++++++')
        procs = UVFits.run_imaging(path_residuals, f'{line}_residuals', imager_executable=IMAGER_EXEC,
                                   script_filename=path_model_to_obs.parent / f'{line}_residuals.imager', device="png")
        print(procs.stderr)
        print('++++++++')
        print(procs.stdout)
        print('\n\n')
    return chi2


def model_chi2_one_param(x):
    return model_chi2(
        tapering_radius=140,
        inner_radius=x,
        log_gas_mass=-2.8,
        temperature_slope=0.5,
        atmosphere_temperature_100au=22,
        midplane_temperature_100au=22,
        inclination_deg=35.18,
        pa_deg=79.19,
        dra=0,
        ddec=0,
        velocity=0.4
    )


if __name__ == '__main__':
    param = np.linspace(100, 300, 4)
    param = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    with multiprocessing.Pool(processes=8) as p:
        chi2 = p.map(model_chi2_one_param, param)
    output = Path('plots')
    output.mkdir(parents=True, exist_ok=True)
    chi2 = np.array(chi2)
    plt.scatter(param, chi2)
    plt.savefig(output / 'chi_innerhole.png')
    # model, _ = model_setup()
