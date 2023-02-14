from pathlib import Path
import logging

import diskchef
diskchef.logging_basic_config()

from diskchef import Line
from diskchef import WilliamsBest100au
from diskchef import SciKitChemistry



from dc_fit import ModelFit, model_in_directory


def main():
    params = [59, -2.65, 35, 11, 0.3]

    model_name = 'DNTau' + ''.join([f'_{str(_param)}' for _param in params])
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


if __name__ == "__main__":
    main()
