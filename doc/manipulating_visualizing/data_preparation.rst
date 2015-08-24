.. _extracting_data:

=========================================================
Data preparation: loading and basic feature extraction
=========================================================

.. contents:: **Contents**
    :local:
    :depth: 1

|

.. topic:: **File names as arguments**

   For most applications of nilearn, it is not necessary to load
   the neuroimaging data oneself.
   Rather than raw data (i.e., numpy arrays),
   most nilearn functions and objects accept file names as
   arguments::

    >>> from nilearn import image
    >>> smoothed_img = image.smooth_img('/home/user/t_map001.nii') # doctest: +SKIP
  
   Nilearn can operate on either file names or `NiftiImage objects
   <http://nipy.org/nibabel/nibabel_images.html>`_. The later
   represent
   the specified nifti files loaded in memory.
   In the context of nilearn, we often use the term 'niimg'
   as abbreviation that denotes either a file name or a
   NiftiImage object. In the example above, the function smooth_img
   returns a NiftiImage object, which can then be readily passed to any
   other nilearn function that accepts niimg arguments.

|

The concept of "masker" objects
=================================

In any analysis, the first step is to load the data.
It is often convenient to apply some basic data
transformations and to turn the data in a 2D (samples x features) matrix,
where the samples could be different time points, and the features derived
from different voxels (e.g., restrict analysis to the ventral visual stream),
regions of interest (e.g., extract local signals from spheres/cubes), or
prespecified networks (e.g., look at data from all voxels of a set of
network nodes). Think of masker objects as swiss army knifes for shaping
the raw neuroimaging data in 3D space into the units of observation
relevant for the research questions at hand.


.. |niimgs| image:: ../images/niimgs.jpg
    :scale: 50%

.. |arrays| image:: ../images/feature_array.jpg
    :scale: 35%

.. |arrow| raw:: html

   <span style="padding: .5em; font-size: 400%">&rarr;</span>

.. centered:: |niimgs|  |arrow|  |arrays|



"masker" objects (found in modules :mod:`nilearn.input_data`) aim at
simplifying these "data folding" steps that often preceed the actual
statistical analysis.

On an advanced note,
the underlying philosophy of these classes is similar to `scikit-learn
<http://scikit-learn.org>`_\ 's
transformers. First, objects are initialized with some parameters guiding
the transformation (unrelated to the data). Then the fit() method
should be called, possibly specifying some data-related
information (such as number of images to process), to perform some
initial computation (e.g., fitting a mask based on the data). Finally,
transform() can be called, with the data as argument, to perform some
computation on data themselves (e.g. extracting time series from images).

Note that the masker objects may not cover all the image transformations
for specific tasks. Users who want to make some specific processing may
have to call low-level functions (see e.g. :mod:`nilearn.signal`,
:mod:`nilearn.masking`).

.. currentmodule:: nilearn.input_data

.. _nifti_masker:

:class:`NiftiMasker`: loading, masking and filtering
=========================================================

This section details how to use the :class:`NiftiMasker` class.
:class:`NiftiMasker` is a
powerful tool to load images and extract voxel signals in the area
defined by the mask. It is designed to apply some basic preprocessing
steps by default with commonly used parameters as defaults. But it is
*very important* to look at your data to see the effects of the
preprocessings and validate them.

In particular, :class:`NiftiMasker` is a `scikit-learn
<http://scikit-learn.org>`_ compliant
transformer so that you can directly plug it into a `scikit-learn
pipeline <http://scikit-learn.org/stable/modules/pipeline.html>`_.

Custom data loading
--------------------

Sometimes, some custom preprocessing of data is necessary. For instance
we can restrict a dataset to the first 100 frames. Below, we load
a resting-state dataset with :func:`fetch_fetch_nyu_rest()
<nilearn.datasets.fetch_nyu_rest>`, restrict it to 100 frames and
build a brand new Nifti-like object to give it to the masker. Although
possible, there is no need to save your data to a file to pass it to a
:class:`NiftiMasker`. Simply use `nibabel
<http://nipy.sourceforge.net/nibabel/>`_ to create a :ref:`Niimg <niimg>`
in memory:


.. literalinclude:: ../../examples/manipulating_visualizing/plot_mask_computation.py
    :start-after: Load NYU resting-state dataset
    :end-before: # To display the background

Controlling how the mask is computed from the data
-----------------------------------------------------

In this tutorial, we show how the masker object can compute a mask
automatically for subsequent statistical analysis.
On some datasets, the default algorithm may however perform poorly.
This is why it is very important to
**always look at your data** before and after feature
engineering using masker objects.

Computing the mask
...................

.. note::
   
    The full example described in this section can be found here:
    :doc:`plot_mask_computation.py <../auto_examples/manipulating_visualizing/plot_mask_computation>`.
    It is also related to this example:
    :doc:`plot_nifti_simple.py <../auto_examples/plot_nifti_simple>`.

If a mask is not specified as an argument,
:class:`NiftiMasker` will try to compute
one from the provided neuroimaging data.
It is *very important* to verify the quality of the generated mask by
visualization. This allows to see whether it
is suitable for your data and intended analyses.
Alternatively, the mask computation parameters can still be modified. See the
:class:`NiftiMasker` documentation for a complete list of mask computation
parameters.

As a first example, we will now automatically build a mask from a dataset.
We will here use the Haxby dataset because it provides the original mask that
we can compare the data-derived mask against.

The first step is to generate a mask with default parameters and visualize it.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_mask_computation.py
    :start-after: # Simple mask extraction from EPI images
    :end-before: # Generate mask with strong opening


.. figure:: ../auto_examples/manipulating_visualizing/images/plot_mask_computation_002.png
    :target: ../auto_examples/plot_mask_computation.html
    :scale: 50%


We can then fine-tune the outline of the mask by increasing the number of
opening steps (*opening=10*) using the `mask_args` argument of the
:class:`NiftiMasker`. This effectively performs erosion and dilation operations
on the outer voxel layers of the mask, which can for example remove remaining
skull parts in the image.

.. literalinclude:: ../../examples/manipulating_visualizing/plot_mask_computation.py
    :start-after: # Generate mask with strong opening
    :end-before: # Generate mask with a high lower cutoff


.. figure:: ../auto_examples/manipulating_visualizing/images/plot_mask_computation_003.png
    :target: ../auto_examples/plot_mask_computation.html
    :scale: 50%


Looking at the :func:`nilearn.masking.compute_epi_mask` called by the
:class:`NiftiMasker` object, we see two interesting parameters:
*lower_cutoff* and *upper_cutoff*. These set the grey-value bounds in
which the masking algorithm will search for its threshold
(0 being the minimum of the image and 1 the maximum). We will here increase
the lower cutoff to enforce selection of those
voxels that appear as bright in the EPI image.


.. literalinclude:: ../../examples/manipulating_visualizing/plot_mask_computation.py
    :start-after: # Generate mask with a high lower cutoff
    :end-before: ################################################################################


.. figure:: ../auto_examples/manipulating_visualizing/images/plot_mask_computation_004.png
    :target: ../auto_examples/plot_mask_computation.html
    :scale: 50%




Common data preparation steps: resampling, smoothing, filtering
-----------------------------------------------------------------

.. seealso::

   If you do not want to use the :class:`NiftiMasker` to perform these
   simple operations on data, note that they can also be manually
   accessed in nilearn such as in
   :ref:`corresponding functions <preprocessing_functions>`.

.. _resampling:

Resampling
..........

:class:`NiftiMasker` and many similar classes enable resampling (recasting
       of images into different resolutions and transformations of brain voxel
       data).
       The resampling procedure takes as input the
       *target_affine* to resample (resize, rotate...) images in order
       to match the spatial configuration defined by the new
       affine (i.e., matrix transforming from voxel space into world space).
       Additionally, a *target_shape* can be used to resize
       images (i.e., cropping or padding with zeros) to match an
       expected data image dimensions (shape composed of x, y, and z).

As a common use case, resampling can be a viable means to
downsample image quality on purpose to increase processing speed
and lower memory consumption of an analysis pipeline.
In fact, certain image viewers (e.g., FSLView) also require images to be
resampled to display overlays.

On an advanced note,
automatic computation of offset and bounding box can be performed by
specifying a 3x3 matrix instead of the 4x4 affine.
In this case, nilearn
computes automatically the translation part of the transformation
matrix (i.e., affine).

.. image:: ../auto_examples/manipulating_visualizing/images/plot_affine_transformation_002.png
    :target: ../auto_examples/plot_affine_transformation.html
    :scale: 36%
.. image:: ../auto_examples/manipulating_visualizing/images/plot_affine_transformation_004.png
    :target: ../auto_examples/plot_affine_transformation.html
    :scale: 36%
.. image:: ../auto_examples/manipulating_visualizing/images/plot_affine_transformation_003.png
    :target: ../auto_examples/plot_affine_transformation.html
    :scale: 36%


.. topic:: **Special case: resampling to a given voxel size**

   Specifying a 3x3 matrix that is diagonal as a target_affine fixes the
   voxel size. For instance to resample to 3x3x3 mm voxels::

    >>> import numpy as np
    >>> target_affine = np.diag((3, 3, 3))

|

.. seealso::

   :func:`nilearn.image.resample_img`


Smoothing
.........

:class:`NiftiMasker` can further be used for local spatial filtering of
the neuroimaging data to make the data more homogeneous and thus account
for inter-individual differences in neuroanatomy.
It is achieved by passing the full-width
half maximum (FWHM; in millimeter scale)
along the x, y, and z image axes by specifying the `smoothing_fwhm` parameter.
For an isotropic filtering, passing a scalar is also possible. The underlying
function handles properly the tricky case of non-cubic voxels by scaling the
given widths appropriately.

.. seealso::

   :func:`nilearn.image.smooth_img`


.. _temporal_filtering:

Temporal Filtering
..................

Rather than optimizing spatial properties of the neuroimaging data,
the user may want to improve aspects of temporal data properties,
before conversion to voxel signals.
:class:`NiftiMasker` can also process voxel signals. Here are the possibilities:

- Confound removal. Two ways of removing confounds are provided. Any linear
  trend can be removed by activating the `detrend` option.
  This accounts for slow (as opposed to abrupt or transient) changes
  in voxel values along a series of brain images that are unrelated to the
  signal of interest (e.g., the neural correlates of cognitive tasks).
  It is not activated
  by default in :class:`NiftiMasker` but is recommended in almost all scenarios.
  More complex confounds can
  be removed by passing them to :meth:`NiftiMasker.transform`. If the
  dataset provides a confounds file, just pass its path to the masker.

- Linear filtering. Low-pass and high-pass filters can be used to remove artifacts.
  It simply removes all voxel values lower or higher than the specified
  parameters, respectively.
  Care has been taken to automatically
  apply this processing to confounds if it appears necessary.

- Normalization. Signals can be normalized (scaled to unit variance) before
  returning them. This is performed by default.

.. topic:: **Exercise**

   You can, more as a training than as an exercise, try to play with
   the parameters in :ref:`example_plot_haxby_simple.py`. Try to enable detrending
   and run the script: does it have a big impact on the result?


.. seealso::

   :func:`nilearn.signal.clean`


Inverse transform: unmasking data
----------------------------------

Once voxel signals have been processed, the result can be visualized as
images after unmasking (masked-reduced data transformed back into
the original whole-brain space). This step is present in almost all
the :ref:`examples <examples-index>` provided in nilearn. Below you will find
an excerpt of :ref:`the example performing Anova-SVM on the Haxby data
<example_decoding_plot_haxby_anova_svm.py>`):

.. literalinclude:: ../../examples/decoding/plot_haxby_anova_svm.py
    :start-after: ### Look at the SVC's discriminating weights
    :end-before: ### Create the figure


.. _region:

Extraction of signals from regions:\  :class:`NiftiLabelsMasker`, :class:`NiftiMapsMasker`.
===========================================================================================

The purpose of :class:`NiftiLabelsMasker` and :class:`NiftiMapsMasker` is to
compute signals from regions containing many voxels. They make it easy to get
these signals once you have an atlas or a parcellation into brain regions.

Regions definition
------------------

Nilearn understands two different ways of defining regions, which are called
labels and maps, handled by :class:`NiftiLabelsMasker` and
:class:`NiftiMapsMasker`, respectively.

- labels: a single region is defined as the set of all the voxels that have a
  common label (e.g., anatomical brain region definitions as integers)
  in the region definition array. The set of
  regions is defined by a single 3D array, containing a voxel-wise
  dictionary of label numbers that denote what
  region a given voxel belongs to. This technique has a big advantage: the
  required memory load is independent of the number of regions, allowing
  for a large number of regions. On the other hand, there are
  several disadvantages: regions cannot spatially overlap
  and are represented in a binary present-nonpresent coding (no weighting).
- maps: a single region is defined as the set of all the voxels that have a
  non-zero weight. A set of regions is thus defined by a set of 3D images (or a
  single 4D image), one 3D image per region (as opposed to all regions in a
  single 3D image such as for labels, cf. above).
  While these defined weighted regions can exhibit spatial
  overlap (as opposed to labels), storage cost scales linearly with the
  number of regions. Handling a large number (e.g., thousands)
  of regions will prove
  difficult with this data transformation of whole-brain voxel data
  into weighted region-wise data.

.. note::

   These usage are illustrated in :ref:`functional_connectomes`

:class:`NiftiLabelsMasker` Usage
---------------------------------

Usage of :class:`NiftiLabelsMasker` is similar to that of
:class:`NiftiMapsMasker`. The main difference is that it requires a labels image
instead of a set of maps as input.

The `background_label` keyword of :class:`NiftiLabelsMasker` deserves
some explanation. The voxels that correspond to the brain or a region
of interest in an fMRI image do not fill the entire
image. Consequently, in the labels image, there must be a label value that
corresponds to "outside" the brain (for which no signal should be
extracted). By default, this label is set to zero in nilearn
(refered to as "background").
Should some non-zero value encoding be necessary, it is
possible to change the background value with the `background_label`
keyword.

:class:`NiftiMapsMasker` Usage
------------------------------

This atlas defines its regions using maps. The path to the corresponding
file is given in the "maps_img" argument. Extracting region signals for
several subjects can be performed like this:

One important thing that happens transparently during the execution of
:meth:`NiftiMasker.fit_transform` is resampling. Initially, the images
and the atlas do typically not have the same shape nor the same affine. Casting
them into the same format is required for successful signal extraction
The keyword argument `resampling_target` specifies which format (i.e.,
dimensions and affine) the data should be resampled to.
See the reference documentation for :class:`NiftiMapsMasker` for every
possible option.
