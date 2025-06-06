import os
import subprocess
import sys

from absl import app
from absl import flags

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory

FLAGS = flags.FLAGS

flags.DEFINE_multi_integer(
    'batch_sizes', [1, 4, 8, 16],
    'Batch sizes to generate StableHLO for.')
flags.DEFINE_multi_integer(
    'iterations', [1, 4, 16],
    'Number of train iterations to include in the StableHLO.')
flags.DEFINE_integer('num_devices', 2,
                     'Number of logical devices for MirroredStrategy.')
flags.DEFINE_string('output_dir', 'output',
                    'Base directory to write StableHLO text files.')


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


def main(_):
    base_dir = FLAGS.output_dir
    os.makedirs(base_dir, exist_ok=True)
    experiments = get_nlp_experiments()

    for batch in FLAGS.batch_sizes:
        for iters in FLAGS.iterations:
            config_dir = os.path.join(base_dir, f"{batch}_{iters}")
            os.makedirs(config_dir, exist_ok=True)
            for exp in experiments:
                outfile = os.path.join(
                    config_dir, exp.replace("/", "_") + ".stablehlo")
                if os.path.exists(outfile):
                    print(f"Skipping {outfile}, already exists")
                    continue
                cmd = [
                    sys.executable,
                    "-m",
                    "official.nlp.tools.dump_stablehlo",
                    f"--experiment={exp}",
                    f"--output={outfile}",
                    f"--num_devices={FLAGS.num_devices}",
                    f"--batch_size={batch}",
                    f"--iterations={iters}",
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app.run(main)
