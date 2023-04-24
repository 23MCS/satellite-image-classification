# Satellite Image Classification

**Remote sensing** technology is an important tool used to examine the earth's surface, and when combined with artificial intelligence in recent years, there has been significant progress in producing **land cover and land use maps**. Traditionally, after remote sensing data is obtained, it is manually interpreted and classified. However, this method is quite time-consuming and costly. Artificial intelligence technologies, especially machine learning techniques, automate this process, resulting in time and cost savings. Artificial intelligence can be very useful for classifying remote sensing data, especially when working with large datasets. For example, machine learning methods can achieve highly successful results for land cover classification using high-resolution remote sensing data.

In addition, remote sensing data can be combined with other data sources to provide more functionality. For example, aerial photographs, satellite imagery, and unmanned aerial vehicles data can be combined and used with artificial intelligence technologies to create land cover and land use maps. In conclusion, remote sensing technology combined with artificial intelligence is a highly effective tool for producing land cover and land use maps, resulting in both time and cost savings. With the development of this technology, more accurate and comprehensive maps can be produced for environmental studies, land use planning, and many other fields.

## DataSet

This GitHub repository contains a classification process using **Sentinel-2** satellite imagery provided as open-source dataset by the European Space Agency. However, this code can also be used for classification processes on any other type of imagery, such as unmanned aerial vehicles or any other type of image.

Before the classification process, the 20m spatial resolution bands of the satellite imagery are resampled to 10m spatial resolution and merged with the 10m resolution bands. This enables the collection of samples for the classes to be used in the classification process using the 10 bands with 10m spatial resolution. Afterwards, the collected samples can be exported either in ".tif" format with a black background (optional) or in ".csv" format containing only the class and feature data of the samples.

## Classification Report

Using this GitHub repository, you can generate a **classification report** that includes model accuracy as well as *precision*, *recall*, and *f-score* values for the classification process.

![classification report](https://github.com/23MCS/satellite-image-classification/blob/main/Classification_Report.png)

## Visualization

If the model accuracy and classification report are satisfactory for the project in the classification process, all pixels of the satellite image are predicted by the model. At the same time, by combining these prediction results with the coordinate data obtained while reading the satellite image, a smoother visualization is obtained.

![classification image](https://github.com/23MCS/satellite-image-classification/blob/main/Classification_Image.png)

## Confusion Matrix

In the final stage of the classification process, the essential and highly informative confusion matrix data, which contains important details, is created and visualized.

![confusion matrix](https://github.com/23MCS/satellite-image-classification/blob/main/Confusiion_Matrix.png)
