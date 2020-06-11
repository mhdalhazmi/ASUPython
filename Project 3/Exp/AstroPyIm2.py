# Astropy Example
# author: allee updated by sdm

from urllib.parse import urlencode          # function to encode internet access
from urllib.request import urlretrieve      # function to retrieve a file from the web

# Astropy package contents required...
from astropy import units as u              # so we can specify arc minutes
from astropy.coordinates import SkyCoord    # get coordinates of objects

# Maybe this works in Spyder?
# from an image processing package
#from IPython.display import Image           # display the image

# I will use this instead...
from PIL import Image                        # display the image

# Set up matplotlib and use a nicer set of plot parameters
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
plt.style.use(astropy_mpl_style)

#telescope_center = SkyCoord.from_name('horsehead nebula')  # use 500 arcmin
#telescope_center = SkyCoord.from_name('andromeda') # use 200 arcmin
telescope_center = SkyCoord.from_name('M51') # use 25 arcmin
#telescope_center = SkyCoord.from_name('andromeda') # use 200 arcmin
#telescope_center = SkyCoord.from_name('polaris') # use 12 arcmin
#telescope_center = SkyCoord.from_name('HCG 7') # use 10 arcmin

print(telescope_center.ra, telescope_center.dec)
print(telescope_center.ra.hour, telescope_center.dec)

# tell the SDSS service how big of a cutout we want
im_size = 25*u.arcmin # get a 12 arcmin square
im_pixels = 4096
cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
query_string = urlencode(dict(ra=telescope_center.ra.deg,
                              dec=telescope_center.dec.deg,
                              width=im_pixels, height=im_pixels,
                              scale=im_size.to(u.arcsec).value/im_pixels))
url = cutoutbaseurl + '?' + query_string

# this downloads the image to your disk
urlretrieve(url, 'telescope.jpg')

# and now we can display it
space = Image.open('telescope.jpg')
space.show()


