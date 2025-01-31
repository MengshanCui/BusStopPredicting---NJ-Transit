
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import QMessageBox


try:
    busfolders = 'BusStop'
    for busfolder in os.listdir(busfolders):
        print(f"{busfolders}/{busfolder}/doesntmatter.txt")
            
except Exception as e:
    print(f"Error reading CSV file: {e}")