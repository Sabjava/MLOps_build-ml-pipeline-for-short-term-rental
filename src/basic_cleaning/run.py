#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    logger.info("Applying price filter")
    price_idx = df['price'].between(args.min_price, args.max_price)

    logger.info("Applying geo boundaries filter")
    geo_idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)

    # Combine filters with AND
    idx = price_idx & geo_idx
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Creating artifact")

    # FIX: Save the file using the actual expected output artifact filename 
    output_filename = args.output_artifact
    df.to_csv(output_filename, index=False)
    
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    
    # FIX: Attach the correctly named file to the artifact
    artifact.add_file(output_filename)

    logger.info("Logging artifact and Uploading data to W&B")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The input file",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The output cleaned file",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact/file",
        required=True
    )
    
    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum Price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum Price",
        required=True
    )

    args = parser.parse_args()

    go(args)