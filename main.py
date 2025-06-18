import argparse
import yaml
import sys
from pathlib import Path

from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train Four-Modal Fusion model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    # Initialize and run the trainer
    trainer = Trainer(config)
    trainer.run()

if __name__ == '__main__':
    main() 