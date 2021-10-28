#!/usr/bin/python2

# import packages
from scripts.utils import *
from noise_induction import NoiseInductor
import json

# load config parameters
with open("config.json") as json_config_file:
    config = json.load(json_config_file)
    
NI = NoiseInductor(config)

# NI.run_noise_induction_model(
#     noise_type='random',
#     output_dir='output/random_error.csv'
# )

NI.run_noise_induction_model(
    noise_type='systematic',
    output_dir='output/systematic_error.csv'
)

# NI.run_denoise_induction_model(
#     output_dir='output/denoise.csv'
# )