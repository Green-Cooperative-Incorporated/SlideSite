import openslide
import os
from openslide.deepzoom import DeepZoomGenerator as dz
from openslide import (OpenSlide, OpenSlideError, OpenSlideUnsupportedFormatError)

#os.add_dll_directory(r"C:\Users\carso\Downloads\openslide-bin-4.0.0.8-windows-x64\openslide-bin-4.0.0.8-windows-x64\bin")
slide = openslide.open_slide(r"C:\Users\carso\Documents\GitHub\SlideSite\TCGA-05-4245-01Z-00-DX1.svs")
thumbnail = slide.get_thumbnail((2048, 2048))
thumbnail.save("thumbnail.png")