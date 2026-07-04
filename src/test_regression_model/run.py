#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")

    # Download test dataset from W&B
    test_local_path = run.use_artifact(args.test_artifact).file()

    # Download model artifact from W&B
    model_artifact = run.use_artifact(args.model_artifact)
    model_local_path = model_artifact.download()

    logger.info("Loading model and data")
    # Load the MLflow sk_pipe model from the downloaded directory
    model = mlflow.sklearn.load_model(model_local_path)
    
    X_test = pd.read_csv(test_local_path)
    y_test = X_test.pop("price")

    logger.info("Evaluating model on test set")
    r_squared = model.score(X_test, y_test)
    
    logger.info(f"Test R2 Score: {r_squared}")
    run.summary["test_r2"] = r_squared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the regression model")
    
    parser.add_argument(
        "--test_artifact", 
        type=str, 
        required=True,
        help="W&B artifact name containing the testing data"
    )
    
    parser.add_argument(
        "--model_artifact", 
        type=str, 
        required=True,
        help="W&B artifact name containing the trained model"
    )

    args = parser.parse_args()
    go(args)