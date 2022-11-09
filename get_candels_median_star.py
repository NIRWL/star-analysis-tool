from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

stars = fits.open('good_candels_stars.fits')[0].data

star_array = stars.reshape(stars.shape[0], stars.shape[1]*stars.shape[2])
med = np.median(star_array, axis=0)
data_diff = star_array - med # Subtract the mean values from the data

# Set pixels that are outliers to the mean value.
for pixel in range(stars.shape[1]*stars.shape[2]):
    d = data_diff[:,pixel]
    std = np.std(d)
    outlier = np.where(np.abs(d) > 3.*std)
    star_array[outlier,pixel] = med[pixel]

meanstar = np.mean(star_array, axis=0)
meanstar = meanstar.reshape(stars.shape[1], stars.shape[2])
disp = (meanstar-meanstar.min()) / (meanstar.max()-meanstar.min())
disp[disp==0] = disp[disp>0].min()
plt.imshow(np.log10(disp), origin='lower')
plt.show()

hdu = fits.PrimaryHDU(meanstar)
hdu.writeto('mean_uds_star.fits')
