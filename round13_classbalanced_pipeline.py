"""Class-specific confidence thresholds with the same teacher and training as round 12."""
import argparse
from round12_pseudo_pipeline import main

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',default='.')
    args=parser.parse_args();args.class_balanced=True
    main(args)

