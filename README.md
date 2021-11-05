# star-analysis-tool

This tool plots a postage stamp image, normalized surface brightness profile, xcorrelation, and whiskers for a selection of stars. The stars should be delivered in a catalog with columns as defined in the code. (This may be updated to be more general in the future.) A fits image must be provided that can be used to do some analysis.

Useage: python star_analysis.py --cat '/Path/to/Catalog.fits' --fname 'Path/to/Fits/Image.fits'
