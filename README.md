# ML-based-Multi-Pool-fitting-of-Z-spectra
Based on the paper published in MRM with the title "Machine-Learning based multi-pool Voigt fitting of chemical exchange saturation transfer (CEST) Z-spectra". The code here presents a framework to set up and train a CatBoost-regressor for the own saturation frequency protocol and targeted solute pools. 

In our application we used 74 saturation frequency points between -20 and 20 ppm and the targeted pools were NOE(-3.6), NOE(-2), water, amine and amide groups located at offset frequencies -3.6, - 2, 0, 2 and 3.6 ppm respectively. These should be changed to the own protocol and targeted pools.

In this framework a catboost-start model is trained on a smaller (with emphasis on small) set of the targeted data which is previously loosely fitted with the defined Least-squared Voigt-model. The aim of this is to ensure high fitting quality of the subsequent LS-fitting to produce training data. The trained catboost-start model is used to generate the starting-values for the following-fits hence ensuring the fitting quality of the training data. The full provided training goes through fitting with LS to generate paired spectra-parameter data for subesquent training of the final model.

It is possible to skip the catboost-start step (with a risk of compromising fitting quality). It is also possible to skip the LS-based fitting process altogether if the user already has paired data available of spectra and the corresponding parameters.

Finally, if the user prefare generating a Multi-pool Lorentzian model it can be done by specifying model = 'lorentzian' when initializing a model-object from the FivePoolModel class. 
