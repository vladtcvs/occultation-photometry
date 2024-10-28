# Occultation-photometry
Tool for analyzing photometry of asteroids star occultations

# Task
We need to analyze tracks of stars during aseroid star occultation to obtain occultation curve

# Components

1. Camera sensor calibration to get pixel e- well overflow parameters (blowing stars)
2. Analyze shape of star track, it can be not straight line
3. Estimate background of track (skyglow)
4. If telescope mount moves during occultation, but not with star speed, discover parameters of such movement by track of not occultated stars
5. Use camera sensor calibration, mount movement calibration, background calibration restore track
6. Deconvolve track with star image (PSF)
7. Obtain occultation curve
   
