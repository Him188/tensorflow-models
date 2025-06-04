import os
import subprocess
import sys

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory


def get_nlp_experiments():
    """Returns a list of experiment names registered for NLP."""
    names = []
    for k, v in exp_factory._REGISTERED_CONFIGS.items():
        if isinstance(v, dict):
            for k2 in v:
                names.append(f"{k}/{k2}")
        else:
            names.append(k)
    return sorted(
        n for n in names if n.startswith(("bert", "electra", "wmt_transformer"))
    )


def main(output_dir="output", num_devices=2):
    os.makedirs(output_dir, exist_ok=True)
    experiments = get_nlp_experiments()
    for exp in experiments:
        outfile = os.path.join(output_dir, exp.replace("/", "_") + ".stablehlo")
        cmd = [
            sys.executable,
            "-m",
            "official.nlp.tools.dump_stablehlo",
            f"--experiment={exp}",
            f"--output={outfile}",
            f"--num_devices={num_devices}",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
