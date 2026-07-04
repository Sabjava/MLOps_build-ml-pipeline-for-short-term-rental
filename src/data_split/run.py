import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def go(args):

    logger.info("Initializing W&B Run for splitting data")
    run = wandb.init(job_type="data_split")

    logger.info("Downloading input artifact from W&B")
    # Fetch the dataset artifact directly from your W&B project dashboard
    artifact = run.use_artifact(args.input_artifact)
    artifact_dir = artifact.download()
    
    # Locate the downloaded file path cleanly inside the local directory
    artifact_path = os.path.join(artifact_dir, "clean_sample.csv")
    if not os.path.exists(artifact_path):
        # Fallback if the file inside the artifact uses a different generic name
        artifact_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

    logger.info("Loading data into dataframe")
    df = pd.read_csv(artifact_path)

    logger.info("Splitting data into trainval and test sets")
    # Stratified split ensures structural proportion matches across groups
    trainval_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by.lower() != "none" else None,
    )

    # Save and upload both outputs back to Weights & Biases
    splits = {
        "trainval_data.csv": trainval_df,
        "test_data.csv": test_df
    }

    for filename, dataframe in splits.items():
        dataframe.to_csv(filename, index=False)
        
        logger.info(f"Logging artifact {filename} to W&B")
        output_artifact = wandb.Artifact(
            name=filename.split(".")[0],
            type="split_data",
            description=f"Output split segment file: {filename}"
        )
        output_artifact.add_file(filename)
        run.log_artifact(output_artifact)
        
        # Clean workspace disk files safely
        if os.path.exists(filename):
            os.remove(filename)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw data into trainval and test groups using W&B Artifacts")

    parser.add_argument(
        "--input_artifact", 
        type=str, 
        required=True,
        help="The full path/tag of the clean input CSV artifact stored in W&B"
    )

    parser.add_argument(
        "--test_size", 
        type=float, 
        required=True,
        help="Percentage size of the test split segment (e.g. 0.2)"
    )

    parser.add_argument(
        "--random_seed", 
        type=int, 
        required=True,
        help="Seed used by the pseudo-random split distribution logic"
    )

    parser.add_argument(
        "--stratify_by", 
        type=str, 
        required=True,
        help="Column to utilize for categorical stratification properties"
    )

    args = parser.parse_args()

    go(args)