# -*- coding: utf-8 -*-

#%%
import numpy as np
from lmfit.models import VoigtModel, LorentzianModel, ConstantModel
from catboost import CatBoostRegressor

class ModelNotTrainedError(Exception):
   
    def __init__(self, message="Model has not been trained"):
        self.message = message
        super().__init__(self.message)
        
class MultiPeakModels:
    def __init__(self, x_values, no_of_peaks = 4, peak_positions = np.r_[-2.8, 2., 3.6], spectral_model = 'voigt',ml_params = 'default'):
        self.x_values = x_values# saturation frequencies in context of Z-spectra
        self.no_of_peaks = no_of_peaks
        self.peak_positions = peak_positions
        self.spectral_model = spectral_model
        self.pars = {"bkg_c":list(),"lz0_center":list() }
        if self.spectral_model == 'lorentzian':    
            for s in range(len(peak_positions)+1):
                self.pars[f"lz{s}_amplitude"] = list() 
                self.pars[f"lz{s}_sigma"] = list()
        elif self.spectral_model == 'voigt':
            for s in range(len(peak_positions)+1):
                self.pars[f"lz{s}_amplitude"] = list() 
                self.pars[f"lz{s}_sigma"] = list()
                self.pars[f"lz{s}_gamma"] = list() 
                
        
        if ml_params == 'default':
            self.ml_params = {'learning_rate': 0.1, 'depth':6, 'loss_function': 'MultiRMSE',
                      'eval_metric': 'MultiRMSE', 'iterations':15000, 'task_type':'GPU',
                      'max_ctr_complexity':15,'l2_leaf_reg':25,'boosting_type': 'Plain',
                      'feature_border_type':'GreedyLogSum','eval_fraction':.05,
                      'early_stopping_rounds':100}
        else:
            self.ml_params = ml_params
        self.ml_model = CatBoostRegressor(**self.ml_params)
        self.ml_model_trained = False
        
                    
    def add_peak(self, prefix, DS = False, center = 0, amplitude=0.5, sigma=0.3, gamma=0.3, p = 0):
        if self.spectral_model == 'voigt':
            peak = VoigtModel(prefix=prefix)
            pars = peak.make_params()
            if DS:
                pars[prefix + 'center'].set(center, min=(-0.3), max=(+0.3))
            else:
                pars[prefix+'center'].expr = f'lz0_center + {p}'
            pars[prefix + 'amplitude'].set(amplitude)
            pars[prefix + 'sigma'].set(sigma, min=0)
            pars[prefix + 'gamma'].set(gamma, min=0,vary = True)
            return peak, pars
        elif self.spectral_model == 'lorentzian':
            peak = LorentzianModel(prefix=prefix)
            pars = peak.make_params()
            if DS:
                pars[prefix + 'center'].set(center, min=(center-0.3), max=(center+0.3))
            else:
                pars[prefix+'center'].expr = f'lz0_center + {p}'
                    
            pars[prefix + 'amplitude'].set(amplitude)
            pars[prefix + 'sigma'].set(sigma, min=0)
            return peak, pars
        else:
            raise ValueError('Only Voigt and Lorentzian line shapes currently available')          

    def sort_params(self, values):
        if self.spectral_model == 'voigt':
            return [[values[0]],values[1:5],values[5:8],values[8:11],values[11:14]]
        elif self.spectral_model == 'lorentzian':
            return [[values[0]],values[1:4],values[4:6],values[6:8],values[8:10]]
        else:
            raise ValueError('Only Voigt and Lorentzian line shapes currently available') 
            
    def create_fit_model_precise(self, peak_p = np.asarray(None)):
        if peak_p.any() == None:    
            model = ConstantModel(prefix='bkg_')
            params = model.make_params(c=0.)
            peak, pars = self.add_peak('lz0_', DS=True,center = 0, amplitude=2.9, sigma=0.7, gamma=0.7)
            model = model + peak
            params.update(pars)
            for i, cen in enumerate(self.peak_positions):
                peak, pars = self.add_peak('lz%d_' % (i+1), DS = False,p=self.peak_positions[i])
                model = model + peak
                params.update(pars)
            return model, params
        
        elif isinstance(peak_p,np.ndarray):
            peak_p = self.sort_params(peak_p)
            
            model = ConstantModel(prefix='bkg_')
            params = model.make_params(c=peak_p[0][0])
            peak, pars = self.add_peak('lz0_', True, amplitude=peak_p[1][0],center = peak_p[1][1], sigma=peak_p[1][2], gamma=peak_p[1][3])
            model = model + peak
            params.update(pars)
            for i, peak in enumerate(peak_p[2:]):
                peak, pars = self.add_peak('lz%d_' % (i+1),DS=False, amplitude= peak[0],sigma = peak[1],gamma = peak[2],p = self.peak_positions[i])
                model = model + peak
                params.update(pars)
            return model, params
        else:
            raise ValueError('peak_p must be None or np.ndarray with shape 14 (voigt) or 10 (Lorentzian)')

    def fitted_spectra(self,labels):
        samples = []
        for label in labels:
            m, p = self.create_fit_model_precise(peak_p = label)
            spectrum = m.eval(params =p, x = self.x_values)
            samples.append(spectrum)
        return np.asarray(samples)
    
    def extract_params(self,params_raw):
        parameters = []    
        
        if self.spectral_model == 'lorentzian':
            for key in self.pars.keys():
                parameters.append(params_raw[key].value)
            return np.asarray(parameters)[[0,2,1,3,4,5,6,7,8,9]]
        elif self.spectral_model == 'voigt':
            for key in self.pars.keys():
                parameters.append(params_raw[key].value)
            return np.asarray(parameters)[[0,2,1,3,4,5,6,7,8,9,10,11,12,13]]
    
    def ls_fit(self, freqdata, Zspectrum, model, params):
        result = model.fit(Zspectrum, params, x=self.x_values, method = 'slsqp')
        
        return self.extract_params(result.params)
    def filter_spectra(self, s,f,l):
        tracker = []
        q = abs(s-f)
        threshold = np.median(q)
        for i,j in enumerate(q):
            if np.mean(j) <threshold:
                tracker.append(i)
        tracker = np.asarray(tracker)        
        return s[tracker], l[tracker]
    
    def train_ml_model(self,train_samples,train_labels,save_path = 'ml_model'):
        self.ml_model.fit(train_samples,train_labels)
        self.ml_model.save_model(save_path)
        print('Training sucessful')
        self.ml_model_trained = True
        return self.ml_model
    def ml_fit(self,spectra):
        if self.ml_model.tree_count_ != None:
            return self.ml_model.predict(spectra)
        else:
            raise ModelNotTrainedError()

