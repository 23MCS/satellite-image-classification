# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:25:29 2023

@author: MCS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from raster2xyz.raster2xyz import Raster2xyz
from osgeo import gdal
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#%%

# Data on the coordinates and pixel digital numbers of the satellite image that has been pre-processed or whose bands have been merged are generated.
# The column names in the DataFrame where the pixel digital numbers values are generated should be organized according to the number of bands used.

rst2xyz_2 = Raster2xyz()
Image_2 = 'Satellite_Image.tif'
rst2xyz_2.translate(Image_2, 'Satellite_Image.csv')
Image_XYZ_2 = pd.read_csv("Satellite_Image.csv")

naip_ds_2 = gdal.Open(Image_2)
nbands_2 = naip_ds_2.RasterCount
data_2 = np.empty((naip_ds_2.RasterXSize*naip_ds_2.RasterYSize, nbands_2))
for i in range(1, nbands_2+1):
    band_2 = naip_ds_2.GetRasterBand(i).ReadAsArray()
    data_2[:, i-1] = band_2.flatten()
    
Image_Pixels_Values = pd.DataFrame(data_2, columns=['Band1', 'Band2','Band3','Band4','Band5','Band6','Band7','Band8','Band9','Band10']).astype(int)


#%% 

# The first and easiest way to process examples of the classes to be used in classification in Python.
# Using GIS software, ".tif" formatted image with a black background is opened, with examples of the classes identified. 

rst2xyz = Raster2xyz()
Image = 'Satellite_Image_Dataset.tif'
rst2xyz.translate(Image, 'Satellite_Image_Dataset.csv')
Image_XYZ = pd.read_csv("Satellite_Image_Dataset.csv")

naip_ds = gdal.Open(Image)
nbands = naip_ds.RasterCount
data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))
for i in range(1, nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()

#%%

# These features are discarded because the background is set to black and there are features that do not represent any class.

data = pd.DataFrame(data)
data.rename(columns = {0:'Class'}, inplace = True)
Dataset = pd.concat([data, Image_Pixels_Values], axis=1)
Dataset = Dataset[Dataset.Class != 0]

#%%

# If your data is already ready and in .csv format, it will be enough to just read the file.

Dataset = pd.read_csv("DataSet.csv",sep=";")

#%%

# The dataset containing only class and feature information is split into a training and a test dataset.

X = Dataset[Dataset.columns[1:]]
Y = Dataset[Dataset.columns[0]]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.3,random_state=1)

#%%

# The data is defined by determining the model to be used in classification.
# Then, model accuracy is obtained using the test data set.

Model_CB = CatBoostClassifier()
Model_CB.fit(X_Train, Y_Train)
print("Accuracy value according to default parameter values of CatBoost algorithm:", Model_CB.score(X_Test, Y_Test))

# Finally, a table containing precision, recall and f-score values for the training of the model is created.

Y_Prediction = Model_CB.predict(X_Test)
Y_Predict = pd.DataFrame(Y_Prediction)
print(classification_report(Y_Test, Y_Predict))

# If the model accuracy is sufficient, all pixels of the satellite image are predicted by the model.

Prediction_CB = Model_CB.predict(Image_Pixels_Values)
Predict_CB = pd.DataFrame(Prediction_CB)

#%%

# The coordinate data of the satellite image and the predicted class values are combined in a DataFrame to obtain a smoother visualization.

CB_Image_XYZ = pd.concat([Image_XYZ, Predict_CB], axis=1, sort=False)
CB_Image_XYZ.columns = ['X','Y','Z','Class']

#%%

# In the final stage of the classification process, the inferred classes are visualized with coordinate data.

fig = plt.figure(figsize=(7,7))
plt.scatter(CB_Image_XYZ['X'] , CB_Image_XYZ['Y'], c=CB_Image_XYZ['Class'], cmap="Spectral")
plt.tight_layout()
plt.title("CatBoost Classifier Image")
plt.show()
fig.savefig('CatBoost.tif')

# The confusion matrix, which provides detailed information about the classification process, is also visualized.

plot_confusion_matrix(Model_CB, X_Test, Y_Test)
plt.title("Confusion Matrix")
plt.show()
