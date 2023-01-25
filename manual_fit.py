import multiprocessing
from pathlib import Path
import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import diskchef

diskchef.logging_basic_config(level=logging.INFO)
from diskchef import SciKitChemistry, Line, WilliamsBest100au, UVFits

from dc_fit import ModelFit, uvs, model_in_directory, lines

# IMAGER_EXEC = "bash -lc imager"
# IMAGER_EXEC = "imager.exe"
IMAGER_EXEC = "imager"


def model_chi2(
        tapering_radius,
        log_gas_mass,
        atmosphere_temperature_100au,
        midplane_temperature_100au,
):
    params = (tapering_radius, log_gas_mass,
              atmosphere_temperature_100au, midplane_temperature_100au)
    root = Path(
        f'fit/{tapering_radius:.1f}_{log_gas_mass:.2f}'
        f'_{atmosphere_temperature_100au:.1f}_{midplane_temperature_100au:.1f}'
    )

    model = model_in_directory(params=params, lines=lines, root=root)
    model.plot(filename=root / "model.png", species1="CO", species2="HCO+")

    chi2 = model.chi2(uvs)  # , dRA=dra, dDec=ddec)
    # with open("chi2.txt", "a") as f:
        # f.write(f"{dra} {ddec} {params} {chi2}\n")

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
    # [50, 10, -2, 0.55, 20, 30, 0.4]
    return model_chi2(
        tapering_radius=50,
        log_gas_mass=-2.3,
        midplane_temperature_100au=19,
        atmosphere_temperature_100au=x,
        # inner_radius=16,  # remove
        # log_gas_mass=-2.8,  # narrow  constraints
        # temperature_slope=0.75,  # maybe fix
        # atmosphere_temperature_100au=35,
        # midplane_temperature_100au=30,  # should also be very narrow [9, 15]`
        # inclination_deg=35.18,
        # pa_deg=79.19,
        # dra=0,  # check these again
        # ddec=0,
        # velocity=0.4
    )


if __name__ == '__main__':
    param = [30]
    PLOT = True
    # param = np.linspace(25, 35, 7)
    with multiprocessing.Pool(processes=8) as p:
        chi2_res = p.map(model_chi2_one_param, param)
    output = Path('plots')
    output.mkdir(parents=True, exist_ok=True)
    if PLOT:
        chi2_res = np.array(chi2_res)
        plt.scatter(param, chi2_res)
        plt.savefig(output / 'chi2.png')
    # model, _ = model_setup()
