# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 04:12:01 2024

@author: MSF-AI
"""

from ML_based_Multi_Peak_Fitting import MultiPeakModels
import numpy as np
#%% Training a CatBoostRegressor for Five pool voigt fitting
params = {'learning_rate': 0.1, 'depth':6, 'loss_function': 'MultiRMSE',
          'eval_metric': 'MultiRMSE', 'iterations':15000, 'task_type':'GPU',
          'max_ctr_complexity':15,'l2_leaf_reg':25,'boosting_type': 'Plain',
          'feature_border_type':'GreedyLogSum','eval_fraction':.05,
          'early_stopping_rounds':100}

saturation_frequencies = '' # saturation frequency protocol

Voigt = MultiPeakModels(x_values = saturation_frequencies,model = 'voigt')
spectra_from_brains = "" # Expected shape None, saturation_frequencies.shape()

params_LS_brains = []
for i,spectrum in enumerate(spectra_from_brains):
    m,p= Voigt.create_fit_model_precise(peak_p = np.asarray(None))
    params_LS_brains.append(Voigt.ls_fit(saturation_frequencies,spectrum,m,p))

params_LS_brains = np.asarray(params_LS_brains)
fits_from_brains = Voigt.fitted_spectra(params_LS_brains)

spectra_from_brains_filtered, params_LS_brains_filtered = Voigt.filter_spectra(spectra_from_brains, fits_from_brains, params_LS_brains)

FPV_ML = CatBoostRegressor(iterations = 250000 ,**params)
FPV_ML.fit(spectra_from_brains_filtered, params_LS_brains_filtered, early_stopping_rounds=5000)

FPV_ML.save_model("FPV_ML")