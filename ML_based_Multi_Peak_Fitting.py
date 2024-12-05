# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from lmfit.models import VoigtModel, LorentzianModel, ConstantModel
from catboost import CatBoostRegressor


class ModelNotTrainedError(Exception):
    """Exception raised when attempting to use an untrained model."""
    
    def __init__(self, message: str = "Model has not been trained"):
        super().__init__(message)
        self.message = message

class MultiPeakModels:
    """
    A class for handling multi-peak spectral models with machine learning capabilities.
    
    Attributes:
        x_values (np.ndarray): Saturation frequencies in the CEST application.
        no_of_peaks (int): Number of peaks to model in the targeted spectra.
        peak_positions (np.ndarray): Predefined peak positions.
        spectral_model (str): Type of spectral model ('voigt' or 'lorentzian').
    """
    
    def __init__(
        self, 
        x_values: np.ndarray, 
        no_of_peaks: int = 4, 
        peak_positions: np.ndarray = np.r_[-2.8, 2., 3.6], 
        spectral_model: str = 'voigt', 
        ml_params: Optional[dict] = None
    ):
        self.x_values = x_values
        self.no_of_peaks = no_of_peaks
        self.peak_positions = peak_positions
        self.spectral_model = spectral_model.lower()
        
        # Validate spectral model
        if self.spectral_model not in ['voigt', 'lorentzian']:
            raise ValueError('Only Voigt and Lorentzian line shapes are currently available')
        
        # Initialize parameters dictionary
        self._pars = self._initialize_parameters()
        self._parsd = self._initialize_parameters_derived()
        
        # Set ML parameters
        self.ml_params = self._get_default_ml_params() if ml_params is None else ml_params
        
        # Initialize ML model
        self.ml_model = CatBoostRegressor(**self.ml_params)
        self.ml_model_trained = False
    
    def _initialize_parameters(self) -> dict:
        """Initialize parameters dictionary based on spectral model."""
        pars = {"bkg_c": [], "lz0_center": []}
        
        # Add parameters based on spectral model
        peak_count = len(self.peak_positions) + 1
        for p in range(peak_count):
            pars[f"lz{p}_amplitude"] = []
            pars[f"lz{p}_sigma"] = []
            
            if self.spectral_model == 'voigt':
                pars[f"lz{p}_gamma"] = []
        return pars
    
    def _initialize_parameters_derived(self) -> dict:
        """Initialize parameters dictionary based on spectral model."""
        pars = {"bkg_c": [], "lz0_center": []}
        
        # Add parameters based on spectral model
        peak_count = len(self.peak_positions) + 1
        for p in range(peak_count):
            pars[f"lz{p}_height"] = []
            pars[f"lz{p}_fwhm"] = []
        return pars
    
    def _get_default_ml_params(self) -> dict:
        """Return default CatBoostRegressor parameters."""
        return {
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'MultiRMSE',
            'eval_metric': 'MultiRMSE',
            'iterations': 15000,
            'task_type': 'GPU',
            'max_ctr_complexity': 15,
            'l2_leaf_reg': 25,
            'boosting_type': 'Plain',
            'feature_border_type': 'GreedyLogSum',
            'eval_fraction': 0.05,
            'early_stopping_rounds': 100
        }
    
    def _add_peak(
        self, 
        prefix: str, 
        DS: bool = False, 
        center: float = 0, 
        amplitude: float = 0.5, 
        sigma: float = 0.3, 
        gamma: float = 0.3, 
        p: float = 0
    ):
        """
        Create a peak model based on the spectral model type.
        
        Args:
            prefix (str): Prefix for the peak parameters.
            DS (bool): Whether it is the center peak or not.
            center (float): Peak center value.
            amplitude (float): Peak amplitude.
            sigma (float): Peak sigma.
            gamma (float): Peak gamma (for Voigt model).
            p (float): Peak position offset if not the center peak.
        
        Returns:
            tuple: Peak model and its parameters.
        """
        if self.spectral_model == 'voigt':
            peak = VoigtModel(prefix=prefix)
            pars = peak.make_params()
            
            # Center parameter setting
            if DS:
                pars[prefix + 'center'].set(center, min=(-0.3), max=(+0.3))
            else:
                pars[prefix+'center'].expr = f'lz0_center + {p}'
            
            # Other parameter settings
            pars[prefix + 'amplitude'].set(amplitude)
            pars[prefix + 'sigma'].set(sigma, min=0)
            pars[prefix + 'gamma'].set(gamma, min=0, vary=True)
            
            return peak, pars
        
        elif self.spectral_model == 'lorentzian':
            peak = LorentzianModel(prefix=prefix)
            pars = peak.make_params()
            
            # Center parameter setting
            if DS:
                pars[prefix + 'center'].set(center, min=(center-0.3), max=(center+0.3))
            else:
                pars[prefix+'center'].expr = f'lz0_center + {p}'
            
            # Other parameter settings
            pars[prefix + 'amplitude'].set(amplitude)
            pars[prefix + 'sigma'].set(sigma, min=0)
            
            return peak, pars
    
    def _create_fit_model_precise(self, peak_p = np.asarray(None)):
        """Generates a model and the corresponding parameters based on the defined
        spectral model and number of peaks"""
        if peak_p.any() == None:    
            model = ConstantModel(prefix='bkg_')
            params = model.make_params(c=0.)
            peak, pars = self._add_peak('lz0_', DS=True, center=0, amplitude=2.9, sigma=0.7, gamma=0.7)
            model = model + peak
            params.update(pars)
            for i, cen in enumerate(self.peak_positions):
                peak, pars = self._add_peak('lz%d_' % (i+1), DS=False, p=self.peak_positions[i])
                model = model + peak
                params.update(pars)
            return model, params
        
        elif isinstance(peak_p, np.ndarray):
            peak_p = self._sort_params(peak_p)
            
            model = ConstantModel(prefix='bkg_')
            params = model.make_params(c=peak_p[0][0])
            if self.spectral_model == 'voigt':
                peak, pars = self._add_peak('lz0_', True, amplitude=peak_p[1][0], center=peak_p[1][1], sigma=peak_p[1][2], gamma=peak_p[1][3])
            elif self.spectral_model == 'lorentzian':
                peak, pars = self._add_peak('lz0_', True, amplitude=peak_p[1][0], center=peak_p[1][1], sigma=peak_p[1][2])
            model = model + peak
            params.update(pars)
            for i, peak in enumerate(peak_p[2:]):
                if self.spectral_model =='voigt':
                    peak, pars = self._add_peak('lz%d_' % (i+1), DS=False, amplitude=peak[0], sigma=peak[1], gamma=peak[2], p=self.peak_positions[i])
                elif self.spectral_model == 'lorentzian':
                    peak, pars = self._add_peak('lz%d_' % (i+1), DS=False, amplitude=peak[0], sigma=peak[1], p=self.peak_positions[i])
                model = model + peak
                params.update(pars)
            return model, params
        else:
            raise ValueError('peak_p must be None or np.ndarray with shape 14 (voigt) or 10 (Lorentzian)')
    
    def _extract_params(self, params_raw, derived_params = False):
        """Internal method to extract parameters from raw parameter object."""
        parameters = []    
        
        if self.spectral_model == 'lorentzian' or derived_params== True:
            for key in self._pars.keys():
                parameters.append(params_raw[key].value)
            return np.asarray(parameters)[[0,2,1,3,4,5,6,7,8,9]]
        elif self.spectral_model == 'voigt':
            for key in self._pars.keys():
                parameters.append(params_raw[key].value)
            return np.asarray(parameters)[[0,2,1,3,4,5,6,7,8,9,10,11,12,13]] 
    
    def _sort_params(self, values, derived_params = False):
        """Internal method to sort parameters based on spectral model."""
        if self.spectral_model == 'voigt' and derived_params ==False:
            return [[values[0]], values[1:5], values[5:8], values[8:11], values[11:14]]
        elif self.spectral_model == 'lorentzian' or derived_params == True:
            return [[values[0]], values[1:4], values[4:6], values[6:8], values[8:10]]
        else:
            raise ValueError('Only Voigt and Lorentzian line shapes currently available')
    
    def _filter_spectra(self, spectra,l=.25,u=.75,scale = 1.5):
        """Internal method to filter spectra based on a threshold."""
        q = spectra[:,0]-spectra[:,-1]
        Q1 = np.quantile(q,l)
        Q3 = np.quantile(q,u)
        IQR = Q3-Q1
        lb = Q1-scale*IQR
        ub = Q3+scale*IQR
        tracker = []
        for i,j in enumerate(q):
           if j <= ub and j>=lb:
               tracker.append(i)       
        return spectra[tracker]
    
    def train_ml_model(
        self, 
        train_samples: np.ndarray, 
        train_labels: np.ndarray, 
        save_path: str = 'ml_model'
    ):
        """
        Train the machine learning model.
        
        Args:
            train_samples (np.ndarray): Training samples.
            train_labels (np.ndarray): Training labels.
            save_path (str): Path to save the model.
        
        Returns:
            CatBoostRegressor: Trained model.
        """
        self.ml_model.fit(train_samples, train_labels)
        self.ml_model.save_model(save_path)
        print('Training successful')
        self.ml_model_trained = True
        return self.ml_model
    
    def ml_fit(self, spectra: np.ndarray) -> np.ndarray:
        """
        Predict using the trained ML model.
        
        Args:
            spectra (np.ndarray): Input spectra to predict.
        
        Raises:
            ModelNotTrainedError: If the model has not been trained.
        
        Returns:
            np.ndarray: Predictions.
        """
        if not self.ml_model_trained:
            raise ModelNotTrainedError()
        
        return self.ml_model.predict(spectra)
    
    def ls_fit(self, Zspectrum, mode = 'raw params'):
        ls_params = []
        model,params = self._create_fit_model_precise()
        
        for i in Zspectrum:
            result = model.fit(i, params, x=self.x_values, method='slsqp')
            if mode == 'raw params':
                ls_params.append(self._extract_params(result.params))
            elif mode == 'derived params':
                ls_params.append(self._extract_params(result.params,derived_params = True))  
        return np.array(ls_params)
    
    def fitted_spectra(self, labels):
        samples = []
        for label in labels:
            m, p = self._create_fit_model_precise(peak_p=label)
            spectrum = m.eval(params=p, x=self.x_values)
            samples.append(spectrum)
        return np.asarray(samples)