''' Kyle's interactive star analysis tool '''
# Visualize the "stars" from a source extracted catalog.
# Create surface brightness profiles and cross-correlation of stars.
# Measure the ellipticity of the stars.
"""
Issues:
- Zoomed axes don't persist.
- Can't select overlapping objects.

"""

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from scipy import interpolate
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from matplotlib.patches import Ellipse
import matplotlib.widgets
import argparse

parser = argparse.ArgumentParser(description='Visualize the stars from a catalog.')
parser.add_argument('--cat', dest="cat", action="store", default="",
                        help="Path to catalog of stars.")
parser.add_argument('--fname', dest="fname", action="store", default="",
                        help="Path to fits file that contains all the stars.")
args = parser.parse_args()


def first_moment(img, gweighted=False, sigma = 3.):
    xsize = img.shape[1]
    ysize = img.shape[0]
    #print 'Your size:', xsize, ysize

    xa = np.array([np.linspace(0, xsize-1, xsize),]*xsize)
    ya = np.array([np.linspace(0, ysize-1, ysize),]*ysize).T

    xa = xa - xsize // 2
    ya = ya - ysize // 2
    if gweighted:
        #Create a map of distance for each pixel from the center pixel
        rmap = np.sqrt(xa**2 + ya**2)
        #Create a Gaussian distribution map.
        gmap = np.exp(-(rmap**2)/(2.*sigma**2))
        gmap = gmap/gmap.sum()

        return (xa*img*gmap).sum() / (img*gmap).sum(), (ya*img*gmap).sum() / (img*gmap).sum()
    else:
        return (xa*img).sum() / img.sum(), (ya*img).sum() / img.sum()


def quadrupole(img, xc, yc, sigma):
    #print(sigma)
    xsize = img.shape[1]
    ysize = img.shape[0]
    #print 'Your size:', xsize, ysize

    xa = np.array([np.linspace(0, xsize-1, xsize),]*xsize)
    ya = np.array([np.linspace(0, ysize-1, ysize),]*ysize).T

    try:
        if (xc.is_integer() == False) or (yc.is_integer() == False):
            print('Warning, the center you have chosen is not an integer pixel!')
    except AttributeError:
        pass

    #Find the offset from the center of the map
    xa = xa - xc
    ya = ya - yc

    #Create a map of distance for each pixel from the center pixel
    rmap = np.sqrt(xa**2 + ya**2)
    #Create a Gaussian distribution map.
    gmap = np.exp(-(rmap**2)/(2.*sigma**2))
    gmap = gmap/gmap.sum()


    #Add the values of the image after Gaussian weighting.
    tot = np.sum(img*gmap)

    q11 = np.sum(xa*xa*img*gmap)/tot
    q22 = np.sum(ya*ya*img*gmap)/tot
    q12 = np.sum(xa*ya*img*gmap)/tot
    q21 = np.sum(ya*xa*img*gmap)/tot

    test = q11*q22 - q12**2

    #Find ellipticities from the quad moments.
    e1 = (q11-q22) / (q11+q22 + 2.*np.sqrt(test))
    e2 = 2.*q12 / (q11+q22 + 2.*np.sqrt(test))

    # This can be considered the size of the object.
    width = np.sqrt(q11 + q22)
    #e = np.sqrt(e1**2 + e2**2)

    return e1, e2, width, q11, q22, q12

def resample_postage(postage, xshift, yshift, s, padding):
    # Define new array that we are going to sample the interpolated image onto.
    x_i = np.linspace(padding, s+padding-1, s) + xshift # Starts from 1 because we want to remove the +1 padding that we added before
    y_i = np.linspace(padding, s+padding-1, s) + yshift
    postage_size = np.linspace(0, postage.shape[0]-1, postage.shape[0]) # Postage size is 23x23 at this point. We will resample it at 21x21
    n_postage = interpolate.interp2d(postage_size, postage_size, postage, kind='cubic')
    n_postage = n_postage(x_i, y_i) # resample at shifted coordinates and 21x21

    return n_postage/np.sum(n_postage)


def radial_profile(center, img):
    xx,yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[0]))
    xx -= center[0]
    yy -= center[1]
    rr = np.sqrt(xx**2+yy**2)

    img /= img[rr<4].sum()

    rmax = img.shape[0] // 2

    binwidth = 0.5
    drs = np.linspace(0,rmax,rmax+1)

    sb = []
    for dr in drs:
        sb.append(np.median(img[(rr < dr+binwidth) & (rr >= dr - binwidth)]))

    return drs, sb

def get_cartesian(xloc, yloc, major_axis, angle):
    #Define the semi-major and minor axis to 1 sigma
    xa = major_axis*np.cos(angle)
    ya = major_axis*np.sin(angle)

    # Determine the end points on the major axis.  The centroid is adjusted for the fit.
    x_ends = [xloc - xa, xloc + xa]
    y_ends = [yloc - ya, yloc + ya]

    return x_ends, y_ends

try:
    mosaic_cat = Table(fits.open(args.cat)[2].data)
except IndexError:
    mosaic_cat = Table(fits.open(args.cat)[1].data)

bm_cat = Table(fits.open('/media/data01/NIRWL/CANDELS_star_catalogs/CANDELS.UDS.v1_1.star_wt_gaia.final.fits')[1].data)
print(bm_cat.colnames)
mosaic = fits.open(args.fname)[0].data # IMAGE
wmos = WCS(fits.open(args.fname)[0].header) # WCS
s = 31 # Postage stamp size
w = s // 2 # Half the postage stamp

from match_cats import match_cats
c1_idx, c2_idx, idx, d2d, d3d = match_cats(mosaic_cat['X_WORLD'], mosaic_cat['Y_WORLD'],bm_cat['RA_CANDELS'],bm_cat['DEC_CANDELS'],d2d_lim=1.)
mosaic_cat = mosaic_cat[c1_idx]
#print(mosaic_cat.colnames)
"""
plt.scatter(mosaic_cat['FLUX_RADIUS']*0.05, mosaic_cat['MAG_AUTO'], color='black', alpha=0.1, s=5)
#plt.scatter(bm_cat['half light radii F160W[pixels]']*0.06, bm_cat['F160W[AB]'])
plt.scatter(stars['FLUX_RADIUS']*0.05, stars['MAG_AUTO'], color='blue')
#plt.xlim(0,5)
plt.ylim(30,15)
plt.xscale('log')
plt.show()

plt.scatter(mosaic_cat['X_WORLD'], mosaic_cat['Y_WORLD'], s=30, c='black')
plt.scatter(bm_cat['RA_CANDELS'], bm_cat['DEC_CANDELS'], s=5, c='red')
plt.show()
sys.exit()
"""

# Create empty lists to store measurements of interest.
profs, mags,rs = [],[],[]
bright, faint = [],[]
e1s,e2s,ws = [],[],[]
xcens,ycens = [],[]
flag = np.zeros(len(mosaic_cat), dtype=bool)
good_stars = np.zeros(len(mosaic_cat), dtype=bool)
postages = np.zeros((len(mosaic_cat), s, s))

padding = 3 # Set the padding for postage because we are going to interpolate and shift it.

for i,star in enumerate(mosaic_cat):
    x, y, = star['XWIN_IMAGE']-1., star['YWIN_IMAGE']-1. # Subtract 1 because SExtractor coordinates start at 1.

    cx = int(round(x,0)) # Use int to remove decimal place.
    cy = int(round(y,0)) # The fraction will be shifted back during resample.

    #Calculate the edges of the postage image
    x1 = cx - w - padding # Lets take a bit larger so that we have room to interpolate
    x2 = cx + w + padding + 1 # +padding for room and +1 more because python slices from x1 to x2-1 .
    y1 = cy - w - padding
    y2 = cy + w + padding + 1

    postage = mosaic[y1:y2, x1:x2] # - star['BACKGROUND'] # if available

    dx,dy = x - cx, y - cy
    post = resample_postage(postage, dx, dy, s, padding) # Resample with the centroid shift.
    post = post.reshape(post.shape[0],post.shape[1])
    postages[i,:,:] = post # Store

    e1, e2, width, q11, q22, q12 = quadrupole(post, post.shape[0]//2, post.shape[1]//2, 4.) # Measure quadrupole moments using a 2. pixel sigma Gaussian weight.
    imax = np.unravel_index(np.argmax(post), post.shape) # Find brightest pixel.
    xcen, ycen = imax[1], imax[0]
    cenx,ceny = first_moment(post, gweighted=False)
    xcens.append(cenx)
    ycens.append(ceny)

    if (star['MAG_AUTO'] > 18) & (star['MAG_AUTO'] < 20) & (ycen == w) & (xcen == w) & (np.sqrt(e1**2+e2**2) < 0.05):
        good_stars[i] = 1 # Mask out faint and shifted stars

    r, sb = radial_profile([post.shape[0]//2, post.shape[1]//2], post)
    sb /= np.linalg.norm(sb) # Normalize the SB profiles so that np.dot(median,median) == 1
    profs.append(sb)
    mags.append(star['MAG_AUTO'])
    rs.append(r)
    e1s.append(e1)
    e2s.append(e2)
    ws.append(width)

profs = np.array(profs)
rs = np.array(rs)
mags = np.array(mags)
e1s = np.array(e1s)
e2s = np.array(e2s)
ws = np.array(ws)
xcens = np.array(xcens)
ycens = np.array(ycens)

e = np.sqrt(e1s**2+e2s**2)
phi = np.arctan2(e2s,e1s)/2.

med_good = np.median(profs[good_stars], axis=0) # Find the median of only the best and brighest stars.
xcorr = np.dot(profs[:,0:5], med_good.T[0:5]) # xcorr of the inner 5 pixels.

''' Set conditions in the keepers variable to put constraints onthe catalog '''
keepers = (xcorr >= 0.98) & (e < 0.05) & (mosaic_cat['MAG_AUTO'] > 18) & (mosaic_cat['MAG_AUTO'] < 23) & (np.abs(xcens) < 0.5) & (np.abs(ycens) < 0.5)# | ((xcorr >= 0.999) & (e < 0.05) & (mosaic_cat['MAG_AUTO'] < 22) & (mosaic_cat['MAG_AUTO'] > 17))
good_stars = mosaic_cat[keepers]
print(len(good_stars), 'good stars remain from', len(mosaic_cat))
test_stars = mosaic_cat[keepers]
test_stars.add_column(e1s[keepers], name='E1')
test_stars.add_column(e2s[keepers], name='E2')
test_stars.add_column(ws[keepers], name='Width')
test_stars.remove_column('MAG_APER')
test_stars.remove_column('MAGERR_APER')
test_stars.write('good_stars.cat', format='ascii')
hdu = fits.PrimaryHDU(postages[keepers])
hdu.writeto('good_candels_stars.fits', overwrite=True)

on = True
if on:

    pa = np.arctan2(e2s,e1s)/2.
    e_ = np.sqrt(e1s**2+e2s**2)

    ratio = (good_stars['XWIN_IMAGE'].max() - good_stars['XWIN_IMAGE'].min()) / (good_stars['YWIN_IMAGE'].max() - good_stars['YWIN_IMAGE'].min())
    fig = plt.figure(figsize=(20,12)) # Set the size of the figure here.
    gs = fig.add_gridspec(2,3)
    ax = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,:2])
    #ax4 = fig.add_subplot(gs[1,2:])

    stamp = ax.imshow(np.zeros([s,s]), origin='lower')

    sbp = ax1.plot(rs.T,profs.T, color='black', alpha=0.1)
    ax1.plot(rs[keepers,:].T,profs[keepers,:].T, color='blue', alpha=0.1)
    sbplot, = ax1.plot(0, 0, color='red')

    ax1.set_yscale('log')
    ax1.set_xlim(0,15)
    ax1.set_ylim(1e-3, 1.)
    ax1.set_xlabel('Radius [Pixels]')
    ax1.set_ylabel('Median Normalized (r<4) Flux in Annulus')
    ax1.set_title('Surface Brightness')

    crosscorr = ax2.scatter(mosaic_cat['MAG_AUTO'], xcorr, color='black')
    ax2.scatter(mosaic_cat['MAG_AUTO'][keepers], xcorr[keepers], color='blue')

    #ax2.scatter(star['MAG_AUTO'], xc, color='red')
    ax2.set_title('Cross-correlation (r<5 pixels)')
    ax2.set_xlabel('F160W Mag')
    ax2.set_ylabel('X-corr')
    ax2.set_ylim(0.9, 1.0)

    xp, yp = get_cartesian(mosaic_cat['XWIN_IMAGE'], mosaic_cat['YWIN_IMAGE'], e*5000, phi)

    for ids,(x0,x1,y0,y1) in enumerate(zip(xp[0], xp[1],yp[0],yp[1])):
        ax3.plot([x0,x1], [y0,y1], color='black', linewidth=1, gid=ids)

    xp, yp = get_cartesian(mosaic_cat['XWIN_IMAGE'][keepers], mosaic_cat['YWIN_IMAGE'][keepers], e[keepers]*5000, phi[keepers])
    for ids,(x0,x1,y0,y1) in enumerate(zip(xp[0], xp[1],yp[0],yp[1])):
        ax3.plot([x0,x1], [y0,y1], color='blue', linewidth=1)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.axis('equal')

    ''' AX4 '''
    #t0 = matplotlib.widgets.TextBox(ax4, 'a: ', 0)
    #t1 = matplotlib.widgets.TextBox(ax4, 'b: ', 1)
    #t2 = matplotlib.widgets.TextBox(ax4, 'b: ', 1)
    #print(t0,t1,t2)
    def update_imshow(ind):
        ax.clear()
        if not ind == None:
            print(ind)
            star = mosaic_cat[ind]

            post = postages[ind,:,:]
            imax = np.unravel_index(np.argmax(post), post.shape) # Find the brightest pixel.
            xcen, ycen = imax[1], imax[0]
            dx,dy = first_moment(post, gweighted=False)

            disp = post - post.min() # Normalized image for plotting.
            disp[disp == 0] = np.median(disp)

            ell = Ellipse((w,w),width=5*(1+e_[ind]*5),height=5, angle=pa[ind]*180/np.pi, facecolor='none', edgecolor='black')
            ax.imshow(np.log10(disp), origin='lower', cmap='jet')
            ax.scatter(xcen,ycen, marker='x', color='white')
            ax.scatter(w+dx,w+dy, marker='o', edgecolor='white', facecolor='black')
            ax.add_artist(ell)
            ax.axvline(w)
            ax.axhline(w)
            ax.text(0,28, 'F160W=%s' % round(star['MAG_AUTO'],1), color='black')
            ax.text(0,27, 'e=%s' % round(e_[ind],3), color='black')
            ax.text(0,26, 'dx=%s,dy=%s' % (round(xcens[ind],3),round(ycens[ind],3)), color='black')
            ax.set_title('Postage Stamp ID: %s' % star['NUMBER'])

    def update_sb(ind):
        #ax1.clear()
        a = ax1.get_lines()
        if len(a) > len(mosaic_cat):
            a.pop(len(mosaic_cat)).remove()
        sbplot, = ax1.plot(rs[ind], profs[ind], color='red')



    def update_xcorr(ind):
        ps = ax2.collections
        for p in ps:
            if p.get_gid() == 100001:
                p.remove()

        star = mosaic_cat[ind]
        xcplot = ax2.scatter(star['MAG_AUTO'], xcorr[ind], color='red', gid=100001)
        return xcplot

    def update_whisker(ind):
        ps = ax3.patches
        for p in ps:
            if p.get_gid() == 100000:
                p.remove()
        star = mosaic_cat[ind]
        ell2 = Ellipse((star['XWIN_IMAGE'], star['YWIN_IMAGE']),width=500*(1+e_[ind]*5),height=500, angle=pa[ind]*180/np.pi, facecolor='none', edgecolor='red', gid=100000)
        whisk = ax3.add_patch(ell2)


    def onclick(event):
        if event.inaxes == ax2:
            cont, gid = crosscorr.contains(event)
            gid_values_list = [*gid.values()]
            gid = int(gid_values_list[0][0])
            if cont:
                #print('Selected %s' % gid)
                update_imshow(gid)
                update_sb(gid)
                update_xcorr(gid)
                update_whisker(gid)
                fig.canvas.draw_idle()

        if event.inaxes == ax3:
            for w in event.inaxes.get_lines():
                if w.contains(event)[0]:
                    gid = w.get_gid()
                    if not gid == None:
                        print('Selected %s' % gid)
                        update_imshow(gid)
                        update_sb(gid)
                        update_xcorr(gid)
                        update_whisker(gid)
                        fig.canvas.draw_idle()




    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

# Turn this on to print out a hard copy of the whiskers and write the stars to a table for future use.

# Make a final whisker plot.
e = e[keepers]
phi = phi[keepers]
e1s = e1s[keepers]
e2s = e2s[keepers]
ratio = (good_stars['XWIN_IMAGE'].max() - good_stars['XWIN_IMAGE'].min()) / (good_stars['YWIN_IMAGE'].max() - good_stars['YWIN_IMAGE'].min())
fig = plt.figure(figsize=(5*ratio,5))
xp, yp = get_cartesian(good_stars['XWIN_IMAGE'], good_stars['YWIN_IMAGE'], e*8000, phi)

for x0,x1,y0,y1 in zip(xp[0], xp[1], yp[0], yp[1]):
    plt.plot([x0,x1], [y0,y1], color='black', linewidth=1)
    #plt.scatter(x0,y0, color='red')
    #plt.scatter(x1,y1, color='blue')

#plt.xlim(0,20000)
#plt.ylim(0,20000)
xp, yp = get_cartesian(plt.xlim()[1]//2, plt.ylim()[1]//2, 0.05*8000, 0.)
plt.plot(xp, yp, color='red', linewidth=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.savefig('star_whisker_CANDELS.jpg', bbox_inches='tight', dpi=200)

# Create a table of the good stars that can be used later to compare with PSF model shapes.
tab = Table([good_stars['NUMBER'], good_stars['X_WORLD'], good_stars['Y_WORLD'], good_stars['XWIN_IMAGE'], good_stars['YWIN_IMAGE'], e1s, e2s], names=['ID', 'RA', 'DEC', 'X', 'Y', 'E1', 'E2'])
print('Saved %s stars.' % len(good_stars))
tab.write('COSMOS_stars_constrained_table.fits', overwrite=True)
