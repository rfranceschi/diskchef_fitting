import logging
import shutil
import signal
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Type, Dict
import os

import astropy.constants as const
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

import diskchef
from diskchef import WilliamsBest100au
from diskchef import NonzeroChemistryWB2014, SciKitChemistry
from diskchef import UltraNestFitter, Parameter
from diskchef import Line
from diskchef import RadMCRTLines, RadMCTherm
from diskchef import UVFits
from diskchef.physics.williams_best import WB100auWithSmoothInnerGap
from diskchef.physics import DustPopulation
from diskchef.dust_opacity import dust_files
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008

diskchef.logging_basic_config(level=logging.WARNING)

radmc_exec = Path(shutil.which('radmc3d'))
if not radmc_exec.is_file():
    raise FileNotFoundError('RADMC3D executable not found')


def sigterm_handler(_signo, _stack_frame):
    """
    Raises SystemExit(1)
    """
    logging.critical("SIGTERM or SIGINT received, trying to wrap the results...")
    sys.exit(1)


"""Catches SIGTERM and SIGINT calling sigterm_handler, making it capturable by try-except-finally"""
signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)

ENV_VARS = dict(
    PYTHONUNBUFFERED=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1
)

for key, value in ENV_VARS.items():
    os.environ[key] = str(value)

uvs = {
    "CO J=2-1": UVFits('observations_test/DNTau/CO_cut.uvfits', sum=False),
    "13CO J=2-1": UVFits('observations_test/DNTau/13CO_cut.uvfits', sum=False),
    "C18O J=2-1": UVFits('observations_test/DNTau/C18O_cut.uvfits', sum=False),
    # "HCO+ J=3-2": UVFits('observations_test/HCO+_cut.uvfits', sum=False),
    # "CO J=2-1": UVFits('observations/s-Line-22-CO_1+D_cut.uvfits', sum=False),
    # "HCO+ J=3-2": UVFits('observations/s-Line-29-HCO+_1+D_cut.uvfits', sum=False),
}

lines = [
    Line(name='CO J=2-1', transition=2, molecule='CO'),
    Line(name='13CO J=2-1', transition=2, molecule='13CO'),
    Line(name='C18O J=2-1', transition=2, molecule='C18O'),
    # Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
    # Line(name='DCO+ J=3-2', transition=3, molecule='DCO+'),
    # Line(name='H13CO+ J=3-2', transition=3, molecule='H13CO+'),
]

# IMAGER_EXEC = "bash -lc imager"
# IMAGER_EXEC = "imager.exe"

chemical_model = SciKitChemistry.load_scikit_model("co_hco+_3params.pkl")
# chemical_model = SciKitChemistry.load_scikit_model("co_hco+_dco+_3params_xray_lum_explored.pkl")

chi2_factor = 1
plt.rc('savefig', dpi=300)
np.seterr('ignore')

DEFAULT_TEMP_DIR: Union[str, Path] = None
"""Default temporaty directory

If None, then looks in os.environ for 'JOB_SHMTMPDIR', then 'JOB_TMPDIR', 'TMPDIR', 'TEMP', 'TMP'
then tempfile.TemporaryDirectory() finds path for temporary directory itself.

If not None, then use as root for temporary directories
"""

if DEFAULT_TEMP_DIR is None:
    for key in 'JOB_SHMTMPDIR', 'JOB_TMPDIR', 'TMPDIR', 'TEMP', 'TMP':
        if key in os.environ.keys():
            DEFAULT_TEMP_DIR = Path(os.environ[key]) / 'diskchef'
            break
    logging.warning("Use %s as root directory for fitting", DEFAULT_TEMP_DIR)
else:
    DEFAULT_TEMP_DIR = Path(DEFAULT_TEMP_DIR)


@dataclass
class ModelFit:
    disk: str
    directory: Union[Path, str] = None
    line_list: List[Line] = None
    physics_class: Type[diskchef.physics.base.PhysicsModel] = WilliamsBest100au
    physics_params: dict = field(default_factory=dict)
    chemistry_class: Type[diskchef.chemistry.base.ChemistryBase] = NonzeroChemistryWB2014
    chemical_params: dict = field(default_factory=dict)
    mstar: u.Msun = 1 * u.Msun
    rstar: u.au = None
    tstar: u.K = None
    inclination: u.deg = 0 * u.deg
    position_angle: u.deg = 0 * u.deg
    distance: u.pc = 100 * u.pc
    velocity: u.km / u.s = 0 * u.km / u.s
    nphot_therm: int = 1e7
    npix: int = 100
    channels: int = 31
    dust_opacity_file: Union[Path, str] = dust_files("diana")[0]
    radial_bins_rt: int = None
    vertical_bins_rt: int = None
    line_window_width: u.km / u.s = 15 * u.km / u.s
    radmc_lines_run_kwargs: dict = field(default_factory=dict)
    mctherm_threads = 1
    camera_refine_criterion: float = 1

    def __post_init__(self):
        self.dust = None
        self.disk_physical_model: diskchef.physics.PhysicsModel = None
        self.disk_chemical_model: diskchef.chemistry.ChemistryModel = None
        self.radmc_model: RadMCRTLines = None
        self.chi2_dict: dict = None
        self._check_radius_temperature()
        self._update_defaults()
        self._initialize_working_dir()
        self.initialize_physics()
        self.initialize_dust()
        self.initialize_chemistry()

    def run_chemistry(self):
        self.disk_physical_model.ionization()
        self.disk_chemical_model.run_chemistry()
        self.disk_chemical_model.table['CO'][self.disk_chemical_model.table['Gas temperature'] > 200 * u.K] = 1e-10
        self.add_co_isotopologs()

    def plot(self, filename="model.png", species1="CO", species2=None):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        self.disk_physical_model.plot_density(axes=ax[0, 0])
        self.disk_physical_model.plot_temperatures(axes=ax[1, 0])
        self.disk_chemical_model.plot_chemistry(species1=species1, species2=species2, axes=ax[0, 1])
        self.disk_chemical_model.plot_absolute_chemistry(species1=species1, species2=species2, axes=ax[1, 1])
        fig.savefig(filename)
        # try:
        self.radmc_model.channel_maps()
        # except Exception as e:
        #     print(e)

    def run_line_radiative_transfer(
            self,
            run_kwargs: dict = None,
            **kwargs
    ):
        """Run line radiative transfer"""
        folder_rt_gas = self.gas_directory

        disk_map = RadMCRTLines(
            chemistry=self.disk_chemical_model, line_list=self.line_list,
            radii_bins=self.radial_bins_rt, theta_bins=self.vertical_bins_rt,
            folder=folder_rt_gas, velocity=self.velocity,
            camera_refine_criterion=self.camera_refine_criterion,
            executable=radmc_exec,
            **kwargs,
        )

        disk_map.create_files(channels_per_line=self.channels, window_width=self.line_window_width)

        disk_map.run(
            inclination=self.inclination,
            position_angle=self.position_angle,
            distance=self.distance,
            npix=self.npix,
            **self.radmc_lines_run_kwargs
        )
        self.radmc_model = disk_map

    def add_co_isotopologs(self, _13c: float = 77, _18o: float = 560):
        self.disk_chemical_model.table['13CO'] = self.disk_chemical_model.table['CO'] / _13c
        self.disk_chemical_model.table['H13CO+'] = self.disk_chemical_model.table['HCO+'] / _13c
        self.disk_chemical_model.table['C18O'] = self.disk_chemical_model.table['CO'] / _18o
        self.disk_chemical_model.table['13C18O'] = self.disk_chemical_model.table['CO'] / (_13c * _18o)
        # remember to fix the next abundance
        # self.disk_chemical_model.table['C17O'] = self.disk_chemical_model.table['CO'] / (
        #         560 * 5)  # http://articles.adsabs.harvard.edu/pdf/1994ARA%26A..32..191W

    def initialize_chemistry(self):
        self.disk_chemical_model = self.chemistry_class(physics=self.disk_physical_model, **self.chemical_params)

    def initialize_dust(self):
        self.dust = DustPopulation(self.dust_opacity_file,
                                   table=self.disk_physical_model.table,
                                   name="Dust")
        self.dust.write_to_table()

    def initialize_physics(self):
        self.disk_physical_model = self.physics_class(**self.physics_params)

    def _update_defaults(self):
        if self.radial_bins_rt is None:
            self.radial_bins_rt = self.physics_params.get("radial_bins", 100)
        if self.vertical_bins_rt is None:
            self.radial_bins_rt = self.physics_params.get("vertical_bins", 100)

    def _initialize_working_dir(self):
        if self.directory is None:
            self.directory = Path(self.disk)
        else:
            self.directory = Path(self.directory)
        self.directory.mkdir(exist_ok=True, parents=True)
        with open(self.directory / "model_description.txt", "w") as fff:
            fff.write(repr(self))
            fff.write("\n")
            fff.write(str(self.__dict__))

    def _check_radius_temperature(self):
        if self.rstar is None and self.tstar is None:
            yb = YorkeBodenheimer2008()
            self.rstar = yb.radius(self.mstar)
            self.tstar = yb.effective_temperature(self.mstar)

    def mctherm(self, threads=None):
        """Run thermal radiative transfer for dust temperature calculation"""
        if threads is None:
            threads = self.mctherm_threads
        folder_rt_dust = self.dust_directory

        self.mctherm_model = RadMCTherm(
            chemistry=self.disk_chemical_model,
            folder=folder_rt_dust,
            nphot_therm=self.nphot_therm,
            star_radius=self.rstar,
            star_effective_temperature=self.tstar
        )

        self.mctherm_model.create_files()
        self.mctherm_model.run(threads=threads)
        self.mctherm_model.read_dust_temperature()

    @property
    def gas_directory(self):
        return self.directory / "radmc_gas"

    @property
    def dust_directory(self):
        return self.directory / "radmc_dust"

    def chi2(self, uvs: Dict[str, UVFits], **kwargs):
        """

        Args:
            uvs: dictionary in a form of {line.name: uvt for line is self.line_list}

        Returns:
            sum of chi2 between uvs and lines
        """

        self.chi2_dict = {
            name: uv.chi2_with(self.gas_directory / f"{name}_image.fits", threads=self.mctherm_threads, **kwargs)
            for name, uv in uvs.items()
            if name in [line.name for line in self.line_list]
        }
        return sum(self.chi2_dict.values())


def model_in_directory(
        params: np.array,
        lines: List[Line],
        root: Union[Path, str] = 'model',
        **kwargs,
) -> ModelFit:
    """
    Create model in directory

    Args:
        params: np.array of parameters for the model
        lines: list of Line objects to make a model for each line
        root:  root directory to create files

    Returns:
        ModelFit instance
    """
    (
        tapering_radius,
        log_gas_mass,
        atmosphere_temperature_100au,
        midplane_temperature_100au,
        temperature_slope,
    ) = params

    # temperature_slope = 0.55
    inner_radius = 11
    # tapering_gamma = 0.75
    velocity = 0.41
    tapering_gamma = 1

    model = ModelFit(
        disk="DN Tau",
        directory=Path(root),
        line_list=lines,
        physics_class=WB100auWithSmoothInnerGapTmidTazzari,
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
            tapering_gamma=tapering_gamma,
            star_mass=0.52 * u.Msun,
            radial_bins=75,
            vertical_bins=75,
            zq=4,
        ),
        chemical_params=dict(
            model=chemical_model
        ),
        inclination=35.18 * u.deg,  # Zhang+ 2019
        position_angle=79.19 * u.deg,  # Zhang+ 2019
        distance=128.22 * u.pc,  # Zhang+ 2019
        velocity=velocity * u.km / u.s,
        npix=200,
        channels=35,
        line_window_width=7 * u.km / u.s,
        **kwargs,
    )
    model.run_chemistry()
    model.run_line_radiative_transfer()
    return model


@dataclass
class WB100auWithSmoothInnerGapTmidTazzari(WB100auWithSmoothInnerGap):

    def gas_temperature(self, r: u.au, z: u.au) -> u.K:
        """Function that returns gas temperature

        at given r,z using the parametrization from
        Williams & Best 2014, Eq. 5-7
        https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59/pdf

        Args:
            r: u.au -- radial distance
            z: u.au -- height
        Return:
            temperature: u.K
        Raises:
            astropy.unit.UnitConversion

            Error if units are not consistent
        """
        temp_midplane = self.midplane_temperature_1au * (r.to(u.au) / u.au) ** (-self.temperature_slope)
        temp_midplane = (temp_midplane**4 + (10 * u.K)**4)**(1/4)
        temp_atmosphere = self.atmosphere_temperature_1au * (r.to(u.au) / u.au) ** (-self.temperature_slope)
        pressure_scalehight = (
                (
                        const.R * temp_midplane * r ** 3 /
                        (const.G * self.star_mass * self.molar_mass)
                ) ** 0.5
        ).to(u.au)
        temperature = u.Quantity(np.zeros_like(z)).value << u.K
        indices_atmosphere = z >= self.zq * pressure_scalehight
        indices_midplane = ~ indices_atmosphere
        temperature[indices_atmosphere] = temp_atmosphere[indices_atmosphere]
        temperature[indices_midplane] = (
                temp_midplane[indices_midplane]
                + (temp_atmosphere[indices_midplane] - temp_midplane[indices_midplane])
                * np.sin((np.pi * z[indices_midplane] / (2 * self.zq * pressure_scalehight[indices_midplane]))
                         .to(u.rad, equivalencies=u.dimensionless_angles())
                         ) ** 4
        )

        return temperature


def my_likelihood(params: np.array) -> float:
    """
    Creates a model for fitting
    Args:
        params: array of parameters for fitting

    Returns:
        -0.5 * chi2 -- loglikelyhood. On exception, writes exception to INFO-level log and returns -inf
    """

    root = DEFAULT_TEMP_DIR
    logging.info("Create files in %s", root)
    root.mkdir(exist_ok=True, parents=True)
    chi2 = np.inf
    try:
        temp_dir = tempfile.TemporaryDirectory(prefix='fit_', dir=root)
        model = model_in_directory(params, lines, root=temp_dir.name)
        chi2 = model.chi2(uvs)
    except MemoryError as e:
        logging.error("Failed with MemoryError! Probably due to high-resolution models and /dev/shm usage")
        logging.error(traceback.format_exc())
        logging.error("If problem persists, set DEFAULT_TEMP_DIR constant in the beginning of dc_fit.py file")
        return -np.inf
    except Exception as e:
        logging.info("Failed with %s", params)
        logging.info(traceback.format_exc())
        return -np.inf
    finally:
        temp_dir.cleanup()
    return -0.5 * chi2 * chi2_factor


def main():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        print("Could not get to MPI!")
    # uvs = {"13CO J=2-1": UVFits('observations/13co.uvfits')}
    # for line in lines:
    #     uv = UVFits("noema_c_uv.pkl")
    #     uv.image_to_visibilities(f'Reference/radmc_gas/{line.name}_image.fits')
    #     uvs[line.name] = uv
    parameters = [
        Parameter(name="R_{c}, au", min=20, max=100, truth=50),
        # Parameter(name="R_{in}, au", min=1, max=40, truth=5),
        Parameter(name="log_{10}(M_{gas}/M_\odot)", min=-3.2, max=-2.0, truth=-2.3),
        # Parameter(name=r"\alpha_{T}", min=0.5, max=0.6, truth=0.55),
        Parameter(name="T_{atm, 100}, K", min=20, max=35, truth=30),
        Parameter(name="T_{mid, 100}, K", min=10, max=25, truth=19),
        # Parameter(name="\gamma", min=0.5, max=1, truth=0.75),
        # Parameter(name="\delta v, km/s", min=0, max=1, truth=0.4),
    ]
    fitter = UltraNestFitter(
        my_likelihood, parameters,
        progress=True,
        storage_backend='hdf5',
        resume=True,
        # run_kwargs={'dlogz': 0.1, 'dKL': 0.1},  # <- very high accuracy
        # run_kwargs={'dlogz': 0.5, 'dKL': 0.5, 'min_num_live_points': 100},  # <- higher accuracy, slower fit
        run_kwargs={'dlogz': 1, 'dKL': 1, 'frac_remain': 0.5, 'Lepsilon': 0.01, 'min_num_live_points': 100},
        # <- lower accuracy, fast fit
    )
    try:
        res = fitter.fit()
    finally:
        if (not fitter.sampler.use_mpi) or (MPI.COMM_WORLD.Get_rank() == 0):
            logging.critical("Saving final results!")
            fitter.save()
            fitter.table.write("table.ecsv")
            fig = fitter.corner()
            fig.savefig("corner.pdf")


if __name__ == '__main__':
    main()
