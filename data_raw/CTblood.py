import pydicom as dicom
import matplotlib.pylab as plt

# specify your image path
image_path = 'data/CTblood/train/ID_000012eaf.dcm'
ds = dicom.dcmread(image_path)
ds.pixel_array.shape
plt.imshow(ds.pixel_array)
