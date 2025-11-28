import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

def setup_logger(name, log_dir="log", level=logging.INFO):
    """
    Sets up a logger with a consistent format.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    log_file = f"{log_dir}/{name}_{int(time.time())}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_pbar(iterable, desc, total=None, leave=False):
    """
    Returns a tqdm progress bar with consistent settings.
    """
    return tqdm(iterable, desc=desc, total=total, unit="step", leave=leave, ncols=100)

def save_results_to_txt(folder_path, file_name, data, fmt='%.6f'):
    """
    Saves data to a txt file.
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    file_path = f"{folder_path}/{file_name}"
    np.savetxt(file_path, data, fmt=fmt)

def log_epoch_stats(logger, epoch, epoch_time=None, loss=None, val_auc=None, val_ap=None, nn_val_auc=None, nn_val_ap=None):
    """
    Logs epoch statistics in a consistent format.
    """
    msg = f"Epoch: {epoch}"
    if epoch_time is not None:
        msg += f" | Time: {epoch_time:.2f}s"
    if loss is not None:
        msg += f" | Loss: {loss:.4f}"
    if val_auc is not None:
        msg += f" | Val AUC: {val_auc:.4f}"
    if val_ap is not None:
        msg += f" | Val AP: {val_ap:.4f}"
    if nn_val_auc is not None:
        msg += f" | New Node Val AUC: {nn_val_auc:.4f}"
    if nn_val_ap is not None:
        msg += f" | New Node Val AP: {nn_val_ap:.4f}"
    logger.info(msg)

def log_test_stats(logger, test_auc, test_ap, nn_test_auc=None, nn_test_ap=None):
    """
    Logs test statistics in a consistent format.
    """
    msg = f"Test Statistics -- Old Nodes AUC: {test_auc:.4f} | AP: {test_ap:.4f}"
    if nn_test_auc is not None and nn_test_ap is not None:
        msg += f" | New Nodes AUC: {nn_test_auc:.4f} | AP: {nn_test_ap:.4f}"
    logger.info(msg)
