# -*- coding: utf-8 -*-
"""
Information
----------
this functions code defines a set of functions used for calculating indices,
as well as displaying images using Matplotlib.
The functions defined in the code are:
1.BandExtractor(im,num_bands): Extracts the bands from the input image using 
gdal's GetRasterBand method and returns a list of bands as 2D numpy arrays.
2.extract_bands(im, num_bands): Extracts the desired number of bands from an 
input image represented as a numpy array.
3.ploting(image, label): Displays an image using Matplotlib and saves it as a PNG file.
4.NBRSWIR(B12, B11): Calculates the Normalized Burn Ratio Short-Wave Infrared 
(NBRSWIR) index using two input bands: Band 12 and Band 11.
5.nbr(B12, B8a): Calculates the Normalized Burn Ratio (NBR) index using two 
input bands: Band 12 and Band 8a.
6.Nbrburnseverity(imgg, output_file): Calculates the NBR burn severity index 
using the NBR index and maps the output values to corresponding classes.
7.NbrPlusburnseverity(imgg, output_file): Calculates the NBR+ burn severity index 
using the NBR+ index and maps the output values to corresponding classes.
8.NDVIclassing(imgg, output_file): Calculates the NDVI classes using the NDVI 
index and maps the output values to corresponding classes.
    
Authors
----------
Sahar Bayati, 
Midya Rostami

Authors Email
----------
saharbyat766@gmail.com,
midyalab@gmail.com 

Authors Website 
----------
http://ecliptic.ir
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings, math
 
warnings.filterwarnings('ignore')


def BandExtractor(im,num_bands):
    """
    Description
    ----------
    The function loops through each band in the image, reads the data for that 
    band using gdal's GetRasterBand method, and appends the band data to the 
    bands list. Finally, the function returns the list of bands.
    
    Parameter(s)
    ----------
    im : gdal.Dataset object
        The input satellite image.
        
    num_bands : int
        The number of bands in the image.
        
    Output(s)
    ----------    
    bands : list of numpy arrays
        A list containing the extracted bands, with each band represented as 
        a 2D numpy array.
    """
    bands = []
    for i in range(num_bands):
        band = im.GetRasterBand(i+1)
        data = band.ReadAsArray()
        bands.append(data)
    return bands 

def extract_bands(im, num_bands):
    """
    Extracts the desired number of bands from an input image.

    Parameters:
    -----------
    im: np.ndarray
        A multi-band image represented as a numpy array. Each band must be a 2D array with the same dimensions.
    num_bands: int
        The number of bands to extract from the input image. This must be less than or equal to the total number of bands in the image.

    Returns:
    -----------
    List[np.ndarray]
        A list containing the extracted bands as numpy arrays. Each band is represented as a 2D array with the same dimensions as the input image.
"""
    bands = BandExtractor(im, num_bands)
    band_list = []
    for band in bands:
        band_list.append(band.astype(np.float32))
    return band_list 
    
def ploting(image,label):
    """
    Description
    ----------
    This function displays an image using matplotlib.

    Parameter(s)
    ----------
    image : ndarray
        Image data as a numpy array

    Output(s)
    ----------    
    None

    """
    plt.figure()
    plt.title(label)
    plt.imshow(image, cmap='jet')
    plt.colorbar()
    plt.savefig(label +'.png',dpi = 800)
    plt.show()

    
#-------------------------------------------------------
#                   NBRSWIr Calculations
#-------------------------------------------------------

def NBRSWIR(B12,B11):
    """
    Description
    ----------
    This function calculates the Normalized Burn Ratio Short-Wave Infrared (NBRSWIR)
    using two input bands: Band 12 and Band 11. 
    The formula for NBRSWIR is (B12 - B11 - 0.02) / (B12 + B11 + 0.1).

    Parameter(s)
    ----------
    B12 : ndarray
        Band 12 data as a numpy array
    B11 : ndarray
        Band 11 data as a numpy array
    Output(s)
    ----------    
    NBRSWIr : ndarray
        NBRSWIR index as a numpy array
    """
    NBRSWIr = (B12-B11-0.02) / (B12+B11+0.1)
    return NBRSWIr


#-------------------------------------------------------
#                   NBR Calculations
#-------------------------------------------------------
def nbr(B12,B8a):
    """
    Description
    ----------
    This function calculates the Normalized Burn Ratio (NBR) index using two input 
    bands: Band 12 and Band 8a.The formula for NBR is (B12 - B8a) / (B12 + B8a).

    Parameter(s)
    ----------
    B12 : ndarray
          Band 12 data as a numpy array
    B8a : ndarray
          Band 8a data as a numpy array

    Output(s)
    ----------    
    nbr : ndarray
        NBR index as a numpy array
    """
    nbr = (B12 - B8a) / (B12 + B8a)
    return nbr

def Nbrburnseverity(imgg, output_file):
    Nbr_burn_severity = np.zeros_like(imgg)
    Nbr_burn_severity[imgg < -0.25] = 0
    Nbr_burn_severity[np.logical_and(imgg >= -0.25 , imgg < -0.1)] = 1
    Nbr_burn_severity[np.logical_and(imgg >= -0.1, imgg < 0.1)] = 2
    Nbr_burn_severity[np.logical_and(imgg >= 0.1, imgg < 0.27)] = 3
    Nbr_burn_severity[np.logical_and(imgg >= 0.27, imgg < 0.44)] = 4
    Nbr_burn_severity[np.logical_and(imgg >= 0.44, imgg < 0.66)] = 5
    class_names = ['High vegetation growth after fire', 
                   'Low growth of post-fire vegetation',
                   'Unburned', 'Burned areas with low severity', 
                   'Burned areas with moderate/low severity', 
                   'Burned areas with moderate/high severity']
    with open(output_file, 'w') as f:
        for i, name in enumerate(class_names):
            num_pixels = np.sum(Nbr_burn_severity == i)
            print(f"Number of pixels in class {name}: {num_pixels}")
            area = num_pixels * 100
            print(f"Area of class {name}: {area} square meters")
            f.write(f"Area of class {name}:\t {area} square meters\n")        
    return Nbr_burn_severity

#-------------------------------------------------------
#                   NDVI Calculations
#-------------------------------------------------------
def Ndvi(B8a,B4):
    """
    Description
    ----------
    This function calculates the Normalized Difference Vegetation Index (NDVI) 
    using two input bands: Band 8a and Band 4. 
    The formula for NDVI is (B8a - B4) / (B8a + B4).

    Parameter(s)
    ----------
    B8a : ndarray
        Band 8a data as a numpy array
    B4 : ndarray
        Band 4 data as a numpy array

    Return(s)
    ----------    
    ndvi : ndarray
         NDVI index as a numpy array
    """
    ndvi = (B8a - B4) / (B8a + B4)
    return ndvi

def NDVIclassing(imgg, output_file):
    NDVIclass = np.zeros_like(imgg)
    NDVIclass[imgg < -0.0] = 0
    NDVIclass[np.logical_and(imgg >= 0.0 , imgg < 0.2)] = 1
    NDVIclass[np.logical_and(imgg >= 0.2, imgg < 0.4)] = 2
    NDVIclass[np.logical_and(imgg >= 0.4, imgg < 0.6)] = 3
    NDVIclass[np.logical_and(imgg >= 0.6, imgg < 0.7)] = 4
    NDVIclass[np.logical_and(imgg >= 0.7, imgg < 1)] = 5
    
    class_names = ['Water or/ bare soil', 
                   'Very low',
                   'Low', 
                   'Moderate Low', 
                   'Moderate High', 
                   'High']
    with open(output_file, 'w') as f:
        for i, name in enumerate(class_names):
            num_pixels = np.sum(NDVIclass == i)
            print(f"Number of pixels in class {name}: {num_pixels}")
            area = num_pixels * 100
            print(f"Area of class {name}: {area} square meters")
            f.write(f"Area of class {name}:\t {area} square meters\n")  
    return NDVIclass


#-------------------------------------------------------
#                   NBR+ Calculations
#-------------------------------------------------------
def NBRPlus(band12,b8a,b3,b2):
    """
    Description
    ----------
    This function calculates the NBR-Plus index using four input bands: Band 12,
    Band 8a, Band 3, and Band 2. 
    The formula for NBR-Plus is (Band 12 - Band 8a - Band 3 - Band 2) / (Band 12 + Band 8a + Band 3 + Band 2).
    
    Parameter(s)
    ----------
    band12 : ndarray
         Band 12 data as a numpy array
    b8a : ndarray
        Band 8a data as a numpy array
    b3 : ndarray
        Band 3 data as a numpy array
    b2 : ndarray
        Band 2 data as a numpy array

    Output(s)
    ----------    
    NBRp : ndarray
        NBR-Plus index as a numpy array
    """
    NBRp = (band12-b8a-b3-b2) / (band12+b8a+b3+b2)
    return NBRp
def NbrPlusburnseverity(imgg, output_file):
    Nbrp_burn_severity = np.zeros_like(imgg)
    Nbrp_burn_severity[imgg < -0.26] = 0
    Nbrp_burn_severity[np.logical_and(imgg >= -0.26 , imgg < -0.15)] = 1
    Nbrp_burn_severity[np.logical_and(imgg >= -0.1, imgg < 0.13)] = 2
    Nbrp_burn_severity[np.logical_and(imgg >= 0.13, imgg < 0.28)] = 3
    Nbrp_burn_severity[np.logical_and(imgg >= 0.28, imgg < 0.50)] = 4
    Nbrp_burn_severity[np.logical_and(imgg >= 0.50, imgg < 1)] = 5
    class_names = ['High vegetation growth after fire', 
                   'Low growth of post-fire vegetation',
                   'Unburned',
                   'Burned areas with low severity', 
                   'Burned areas with moderate/low severity', 
                   'Burned areas with moderate/high severity']
    with open(output_file, 'w') as f:
        for i, name in enumerate(class_names):
            num_pixels = np.sum(Nbrp_burn_severity == i)
            print(f"Number of pixels in class {name}: {num_pixels}")
            area = num_pixels * 100
            print(f"Area of class {name}: {area} square meters")
            f.write(f"Area of class {name}:\t {area} square meters\n")    
    return Nbrp_burn_severity


#-------------------------------------------------------
#                   LST Calculations
#-------------------------------------------------------

def sentinelLST(B4,B10,B8):
    """
    Description
    ----------
    Calculate land surface temperature using the Split-Window algorithm
    Parameter(s)
    ----------
    band10 : ndarray
         Band 10 data as a numpy array
    b8 : ndarray
        Band 8 data as a numpy array
    b4 : ndarray
        Band 4 data as a numpy array

    Output(s)
    ----------    
    LST : ndarray
        LST as a numpy array
    """
    a = B4
    b = B8
    c = B10
    #LST = c.divide(10).divide(a.add(b).multiply(0.5).multiply(0.0001)).divide(1 + (0.00115 * c.divide(10).divide(1.4388 / 1.2) * (a.add(b).multiply(0.5).multiply(0.0001)).log())).subtract(273.15)
    #lst = (c / (1 + (0.00115 * (c / 1.4388) / 1.2) * np.log(a.add(b).multiply(0.5).multiply(0.0001)))) - 273.15
    Ep = (a + b) * 0.5 * 0.0001
    Tb = c * 0.1
    LST = (Tb / (1 + (0.00115 * (Tb / 1.4388) / 1.2) * np.log(Ep))) - 273.15
    LST = (Tb / (1 + (0.00115 * (Tb / 1.4388) / 1.2) * np.log(Ep))) - 273.15

    return LST        


