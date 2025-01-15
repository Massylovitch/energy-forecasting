import json
import fire
import wandb
from training_pipeline import utils
from training_pipeline.settings import SETTINGS, OUTPUT_DIR


def upload(sweep_id=None):

    if sweep_id is None:
        last_sweep_metadata = utils.load_json("last_sweep_metadata.json")
        sweep_id = last_sweep_metadata["sweep_id"]

    api = wandb.Api()
    sweep = api.sweep(
        f"{SETTINGS['WANDB_ENTITY']}/{SETTINGS['WANDB_PROJECT']}/{sweep_id}"
    )
    best_run = sweep.best_run()

    with utils.init_wandb_run(
        name="best_experiment",
        job_type="hpo",
        group="train",
        run_id=best_run.id,
        resume="must",
    ) as run:
        run.use_artifact("config:latest")

        best_config = dict(run.config)

        config_path = OUTPUT_DIR / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=4)

        artifact = wandb.Artifact(
            name="best_config",
            type="model",
            metadata={"results": {"validation": dict(run.summary["validation"])}},
        )
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)

        run.finish()


if __name__ == "__main__":
    fire.Fire(upload)
