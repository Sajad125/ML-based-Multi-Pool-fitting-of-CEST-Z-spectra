# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 04:12:01 2024

@author: MSF-AI
"""

from ML_based_Multi_Peak_Fitting import MultiPeakModels, Evaluation
import numpy as np
#%% Training a CatBoostRegressor for Five pool voigt fitting

x_values = np.load('cest_freq.npy') # saturation frequency protocol in this case -20 to 20 ppm 
Zspectra_raw = np.load('example_spectra.npy')  # Expected shape None, saturation_frequencies.shape()

Voigt = MultiPeakModels(x_values = x_values,spectral_model = 'voigt')

#Removing outliers
Zspectra = Voigt._filter_spectra(Zspectra_raw) # Tukey method applied to remove spectra non compliant with the spectal shapes
params_LS = Voigt.ls_fit(Zspectra)

ml_model = Voigt.train_ml_model(Zspectra,params_LS,'trained_model')

#%% Eval

test_s = np.load('test_s.npy')
test_l = np.load('test_l.npy')
test_p = ml_model.predict(test_s)

evaluate = Evaluation(x_values,test_s,test_p,test_l)

print(evaluate.R_squared)
evaluate.plot_spectra() # Move though the fitted spectra
evaluate.create_bland_altman_subplots(mode = 'derived params') # Move through the difference plots of each parameter