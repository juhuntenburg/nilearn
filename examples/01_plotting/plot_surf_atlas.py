"""
Demonstrate basic loading and plotting of a cortical surface atlas
===================================================================
The Destrieux parcellation (Destrieux et al, 2010) in fsaverage5 space as
distributed with Freesurfer is used as the chosen atlas.

The :func:`nilearn.plotting.plot_surf_roi` function is used
to plot the parcellation on the pial surface.

See :ref:`plotting` for more details.

References
----------

Destrieux et al, (2010). Automatic parcellation of human cortical gyri and
sulci using standard anatomical nomenclature. NeuroImage, 53, 1.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.06.010.
"""

###############################################################################
from nilearn import plotting
from nilearn import datasets
import nibabel as nb

###############################################################################
# Destrieux parcellation left hemisphere in fsaverage5 space
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation = destrieux_atlas['map_left']

# Retrieve fsaverage data
fsaverage = datasets.fetch_surf_fsaverage5()

# Fsaverage5 left hemisphere surface mesh files
fsaverage5_pial = fsaverage['pial_left']
fsaverage5_inflated = fsaverage['infl_left']
sulcal_depth_map = fsaverage['sulc_left']

###############################################################################
# Display Destrieux parcellation on fsaverage5 surface
plotting.plot_surf_roi(fsaverage5_pial, roi_map=parcellation,
                       hemi='left', view='lateral', bg_map=sulcal_depth_map,
                       bg_on_data=True, darkness=.5, cmap='gist_ncar')

# Display Destrieux parcellation on inflated fsaverage5 surface
plotting.plot_surf_roi(fsaverage5_inflated, roi_map=parcellation,
                       hemi='left', view='lateral', bg_map=sulcal_depth_map,
                       bg_on_data=True, darkness=.5, cmap='gist_ncar')

plotting.show()
