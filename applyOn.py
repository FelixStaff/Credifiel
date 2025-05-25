import torch
from models.transaction_classifier import Transaction
from pipeline.timeseries_pipeline import pipeline
import random
from readData import PreprocesadorCobros
from readData import convertir_a_tensor
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report