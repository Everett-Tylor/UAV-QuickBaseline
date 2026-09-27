"""Single DINOv3, offline ClassMix and moderate true-barren oversampling."""
import argparse
from round12_pseudo_pipeline import main

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='.')
    a=p.parse_args();a.classmix=True
    main(a)
