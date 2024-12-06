# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
from lmfit.models import VoigtModel, LorentzianModel, ConstantModel
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.special as scs


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
    
    def fitted_spectra(self, labels):
        samples = []
        for label in labels:
            m, p = self._create_fit_model_precise(peak_p=label)
            spectrum = m.eval(params=p, x=self.x_values)
            samples.append(spectrum)
        return np.asarray(samples)
    
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
    

class Evaluation:
    """
    A class for evaluating multi-peak spectral by the MultiPeakModels class.

    Attributes:
        x_values (np.ndarray): Saturation frequencies in the CEST application.
        test_samples (np.ndarray): Test spectra.
        predictions (np.ndarray): The ML-fitted parameters for the test spectra.
        test_labels (np.ndarray): The ground truth parameters, if provided will allow Bland Altman analysis.
        spectral_model (str): Type of spectral model ('voigt' or 'lorentzian').
    """

    def __init__(self, x_values, test_samples, predictions, test_labels=None, no_of_peaks=4,
                 peak_positions=np.r_[-2.8, 2., 3.6], spectral_model='voigt'):
        self._x_values = x_values
        self._test_samples = test_samples
        self.spectral_model = spectral_model

        self._multipeakmodel = MultiPeakModels(x_values, no_of_peaks=no_of_peaks,
                                                peak_positions=peak_positions, 
                                                spectral_model=spectral_model)
        self._fits = self._multipeakmodel.fitted_spectra(predictions)
        self.R_squared = self._calculate_R_squared(test_samples, self._fits)

        self._test_preds = predictions
        self._test_labels = test_labels
        
        if test_labels is not None:
            self._prepare_test_labels(test_labels, predictions)

    def _prepare_test_labels(self, test_labels, predictions):
        """Calculate derived parameters for test labels and filter outliers."""
        derived_labels = self._derived_params(test_labels)
        derived_preds = self._derived_params(predictions)
        outliers = self._outlier_removal(derived_preds)

        self._test_labelsd = derived_labels[outliers]
        self._test_predsd = derived_preds[outliers]

    def plot_fits(self):
        """Plot fitted spectra with the ability to navigate through samples."""
        fig, ax = plt.subplots()
        current_index = 0
        scatter = ax.scatter(self._x_values, self._test_samples[current_index], label='Sample')
        line, = ax.plot(self._x_values, self._fits[current_index], label='Fit')

        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        ax.legend()

        def on_key(event):
            nonlocal current_index
            if event.key == '1':
                current_index = max(0, current_index - 1)
            elif event.key == '2':
                current_index = min(len(self._test_samples) - 1, current_index + 1)

            scatter.set_offsets(list(zip(self._x_values, self._test_samples[current_index])))
            line.set_ydata(self._fits[current_index])
            ax.set_title(f'Current Index: {current_index + 1}')
            fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    def create_bland_altman_paramwise(self, mode='raw params'):
        """Create paramwise Bland-Altman plots."""
        if self._test_labels is None:
            raise Exception('Ground truth labels need to be provided for Bland Altman analysis')

        if mode == 'raw params':
            labels = self._test_labels
            preds = self._test_preds
        elif mode == 'derived params':
            labels = self._test_labelsd
            preds = self._test_predsd
        else:
            raise ValueError('Mode must be "raw params" or "derived params"')

        current_subplot = 0
        num_subplots = labels.shape[1]
        fig, ax = plt.subplots(figsize=(10, 6))

        def update_subplot(index):
            ax.clear()
            data1 = labels[:, index]
            data2 = preds[:, index]

            mean = np.mean([data1, data2], axis=0)
            diff = data1 - data2
            md = np.mean(diff)
            sd = np.std(diff, axis=0)

            ax.scatter(mean, diff, s=2, label='Data Points')
            ax.axhline(md, color='red', linestyle='--', label='Mean Difference')
            ax.axhline(md + 1.96 * sd, color='black', linestyle='--', label='+1.96 SD')
            ax.axhline(md - 1.96 * sd, color='black', linestyle='--', label='-1.96 SD')

            ax.set_ylim([md - 8 * sd, md + 8 * sd])
            ax.set_xlim([np.median(mean) - 2.5 * np.std(mean), np.median(mean) + 2.5 * np.std(mean)])
            ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=3))
            ax.set_yticks(np.linspace(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1], num=5))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.set_title(f"{('Voigt' if self.spectral_model == 'voigt' else 'Lorentzian')} param: {index}")
            ax.legend(loc='upper right')  # Add legend

            fig.text(0.5, 0.015, "Mean of ML and LS parameter", ha='center', fontsize=20)
            fig.text(0.03, 0.5, "Difference of ML and LS parameter", va='center', rotation='vertical', fontsize=20)
            plt.draw()

        def on_key(event):
            nonlocal current_subplot
            if event.key == '1':
                current_subplot = max(0, current_subplot - 1)
            elif event.key == '2':
                current_subplot = min(num_subplots - 1, current_subplot + 1)
            update_subplot(current_subplot)

        update_subplot(current_subplot)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    def create_bland_altman_summarized(self, mode='raw params'):
        """Create summarized Bland-Altman plots."""
        if self._test_labels is None:
            raise Exception('Ground truth labels need to be provided for Bland Altman analysis')

        labels = self._test_labels if mode == 'raw params' else self._test_labelsd
        preds = self._test_preds if mode == 'raw params' else self._test_predsd
        num_subplots = labels.shape[1]

        num_cols = 4
        num_rows = (num_subplots + num_cols - 1) // num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

        for i in range(num_subplots):
            data1 = labels[:, i]
            data2 = preds[:, i]
            mean = np.mean([data1, data2], axis=0)
            diff = data1 - data2
            md = np.mean(diff)
            sd = np.std(diff, axis=0)

            row, col = divmod(i, num_cols)
            ax = axs[row, col]
            ax.scatter(mean, diff,color='blue', s=2)
            ax.axhline(md, color='red', linestyle='--')
            ax.axhline(md + 1.96 * sd, color='black', linestyle='--')
            ax.axhline(md - 1.96 * sd, color='black', linestyle='--')
            if self.spectral_model == 'voigt':
                ax.text(0.02, 0.98, f"Voigt param:{i}", transform=ax.transAxes, fontsize=20, va='top')
            else:
                ax.text(0.02, 0.98, f"Lorentzian param: {i}", transform=ax.transAxes, fontsize=20, va='top')
            ax.set_ylim([md - 8 * sd, md + 8 * sd])
            ax.set_xlim([np.median(mean) - 2.5 * np.std(mean), np.median(mean) + 2.5 * np.std(mean)])

            ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=3))
            ax.set_yticks(np.linspace(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1], num=5))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

        ax.scatter(mean, diff,color='blue', s=2, label='Data Points')        
        ax.axhline(md, color='red', linestyle='--', label='Mean Difference')
        ax.axhline(md + 1.96 * sd, color='black', linestyle='--', label='+1.96 SD')
        ax.axhline(md - 1.96 * sd, color='black', linestyle='--', label='-1.96 SD')
        ax = axs.flatten()[num_subplots:]  # Hide any unused subplots
        for a in ax:
            a.axis("off")

        # Legend for the last subplot
        
        fig.legend(loc = 'lower right', fontsize=12, bbox_to_anchor=(0.65, 0.10))

        fig.text(0.5, 0.05, "Mean of ML and LS parameter", ha='center', fontsize=20)
        fig.text(0.04, 0.5, "Difference of ML and LS parameter", va='center', rotation='vertical', fontsize=20)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()

    def _calculate_R_squared(self, experimental_data, fitted_data):
        """Calculate the R-squared value."""
        ss_total = np.sum((experimental_data - np.mean(experimental_data)) ** 2)
        ss_residual = self._calculate_RSS(experimental_data, fitted_data)
        return 1 - (ss_residual / ss_total)

    def _calculate_RSS(self, experimental_data, fitted_data):
        """Calculate the residual sum of squares."""
        residuals = experimental_data - fitted_data
        return np.sum(residuals ** 2)

    def _amptoheight(self, lz_amp, lz_gamma, lz_sigma=None):
        """Convert amplitude to height."""
        if self.spectral_model == 'lorentzian':
            return 0.3183099 * lz_amp / np.maximum(1e-15, lz_gamma)
        elif self.spectral_model == 'voigt':
            return (lz_amp / np.maximum(1e-15, lz_sigma * np.sqrt(2 * np.pi))) * np.real(
                scs.wofz((1j * lz_gamma) / np.maximum(1e-15, lz_sigma * np.sqrt(2)))
            )

    def _lwtofwhm(self, lz_sigma=None, lz_gamma=None):
        """Convert Lorentzian parameters to full width at half maximum (FWHM)."""
        if self.spectral_model == 'lorentzian':
            return 2.0 * lz_gamma
        elif self.spectral_model == 'voigt':
            return 1.0692 * lz_gamma + np.sqrt(0.8664 * lz_gamma ** 2 + 5.545083 * lz_sigma ** 2)

    def _derived_params(self, p):
        """Calculate derived parameters from fits."""
        params = np.zeros([p.shape[0], 10])
        params[:, 0] = p[:, 0]
        params[:, 1] = self._amptoheight(-1 * p[:, 1], p[:, 4], p[:, 3]) if self.spectral_model == 'voigt' else self._amptoheight(-1 * p[:, 1], p[:, 3])
        params[:, 2] = p[:, 2]
        params[:, 3] = self._lwtofwhm(p[:, 3], p[:, 4]) if self.spectral_model == 'voigt' else self._lwtofwhm(lz_gamma=p[:, 3])
        params[:, 4] = self._amptoheight(-1 * p[:, 5], p[:, 7], p[:, 6]) if self.spectral_model == 'voigt' else self._amptoheight(-1 * p[:, 4], p[:, 5])
        params[:, 5] = self._lwtofwhm(p[:, 6], p[:, 7]) if self.spectral_model == 'voigt' else self._lwtofwhm(lz_gamma=p[:, 5])
        params[:, 6] = self._amptoheight(-1 * p[:, 8], p[:, 10], p[:, 9]) if self.spectral_model == 'voigt' else self._amptoheight(-1 * p[:, 6], p[:, 7])
        params[:, 7] = self._lwtofwhm(p[:, 9], p[:, 10]) if self.spectral_model == 'voigt' else self._lwtofwhm(lz_gamma=p[:, 7])
        params[:, 8] = self._amptoheight(-1 * p[:, 11], p[:, 13], p[:, 12]) if self.spectral_model == 'voigt' else self._amptoheight(-1 * p[:, 8], p[:, 9])
        params[:, 9] = self._lwtofwhm(p[:, 12], p[:, 13]) if self.spectral_model == 'voigt' else self._lwtofwhm(lz_gamma=p[:, 9])

        return params

    def _outlier_removal(self, samples, lower_quantile=0.25, upper_quantile=0.75, params='all'):
        """Remove outlier samples based on the IQR method."""
        tracker = []
        param_indices = range(samples.shape[1]) if params == 'all' else params
        
        for i in param_indices:
            q = samples[:, i]
            q1, q3 = np.quantile(q, [lower_quantile, upper_quantile])
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
            outliers = [j for j, value in enumerate(q) if value < lb or value > ub]
            tracker.extend(outliers)
        
        return list(set(range(len(samples))) - set(tracker))