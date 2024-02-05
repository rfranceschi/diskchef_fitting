from pathlib import Path

import diskchef
from diskchef import Line, UVFits

from dc_fit import model_in_directory

diskchef.logging_basic_config()


def main():
    params = [100, -2.9, 10, 2, 0.55]

    disk = 'DNTau'
    model_name = f'{disk}_test' + ''.join([f'_{str(_param)}' for _param in params])
    root = Path('Reference') / model_name

    lines = [
        Line(name='CO J=2-1', transition=2, molecule='CO'),
        Line(name='13CO J=2-1', transition=2, molecule='13CO'),
        Line(name='C18O J=2-1', transition=2, molecule='C18O'),
        # Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
        # Line(name='DCO+ J=3-2', transition=3, molecule='DCO+'),
        # Line(name='H13CO+ J=3-2', transition=3, molecule='H13CO+'),
    ]

    model = model_in_directory(params, lines=lines, root=root)
    model.plot(filename=root / "model.png", species1="13CO", species2="CO")

    uvs = {
        "CO J=2-1": UVFits(f'observations_new/{disk}/CO_cut.uvfits', sum=False),
        "13CO J=2-1": UVFits(f'observations_new/{disk}/13CO_cut.uvfits', sum=False),
        "C18O J=2-1": UVFits(f'observations_new/{disk}/C18O_cut.uvfits', sum=False),
        # "CO J=2-1": UVFits('observations/DNTau_old/CO_cut.uvfits', sum=False),
        # "13CO J=2-1": UVFits('observations/DNTau_old/13CO_cut.uvfits', sum=False),
        # "C18O J=2-1": UVFits('observations/DNTau_old/C18O_cut.uvfits', sum=False),
        # "HCO+ J=3-2": UVFits('observations/HCO+_cut.uvfits', sum=False),
        # "CO J=2-1": UVFits('observations/s-Line-22-CO_1+D_cut.uvfits', sum=False),
        # "HCO+ J=3-2": UVFits('observations/s-Line-29-HCO+_1+D_cut.uvfits', sum=False),
    }

    chi2 = model.chi2(uvs)
    print(model_name)
    print(f'{chi2:.2e}')


if __name__ == "__main__":
    main()
