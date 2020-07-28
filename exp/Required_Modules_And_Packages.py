# importing necessary modules

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from Precision_Module import Precision1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import aidrtokenize
from sklearn.metrics import classification_report

from pathlib import Path
import os
import torch
import torch.optim as optim
import random
import tarfile
import zipfile

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import AdamW
from functools import partial