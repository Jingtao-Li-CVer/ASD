import numpy as np
from osgeo import gdal


def read_img(img_path: str):
    """
    Read imagery as ndarray
    :param img_path:
    :param gdal_read:
    :return:
    """
    dataset = gdal.Open(img_path)
    w, h = dataset.RasterXSize, dataset.RasterYSize
    img = dataset.ReadAsArray(0, 0, w, h)
    if len(img.shape) == 3:
        img = np.transpose(img, axes=(1, 2, 0))  # [c,h,w]->[h,w,c]
    return img


def write_img(img: np.ndarray, save_path: str):
    """
    Save ndarray as imagery
    :param img:
    :param save_path:
    :param gdal_write: 
    :return:
    """
    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img.shape) == 3:
        img = np.transpose(img, axes=(2, 0, 1))  # [h,w,c]->[c,h,w]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)

    img_bands, img_height, img_width = img.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, int(img_width), int(img_height), int(img_bands), datatype)
    for i in range(img_bands):
        dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset

if __name__ == "__main__":
    a = 1