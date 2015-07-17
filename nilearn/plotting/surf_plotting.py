"""
Functions for surface visualization.
Only matplotlib is required.
"""

# display figure? return figure?
# scaling of figure
# plotting roi
# reuse _get_plot_surf_params
# display colorbar
# gifti files?

from nilearn._utils.compat import _basestring
from .img_plotting import _get_plot_surf_params

# Import libraries
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D


# function to figure out datatype and load data
def check_surf_data(filename):
    if (filename.endswith('nii') or filename.endswith('nii.gz') or
            filename.endswith('mgz')):
        data = np.squeeze(nib.load(filename).get_data())
    elif (filename.endswith('curv') or filename.endswith('sulc') or
            filename.endswith('thickness')):
        data = nib.freesurfer.io.read_morph_data(filename)
    elif filename.endswith('annot'):
        data = nib.freesurfer.io.read_annot(filename)[0]
    elif filename.endswith('label'):
        data = nib.freesurfer.io.read_label(filename)
    else:
        data = None
    return data


def plot_surf(mesh, hemi, stat_map=None, bg_map=None, threshold=None,
              view='lateral', cmap='RdBu_r', alpha='auto', vmax=None,
              symmetric_cbar="auto", output_file=None, **kwargs):

    """ Plotting of surfaces with optional background and stats map
    """

    # load mesh and derive axes limits
    coords, faces = nib.freesurfer.io.read_geometry(mesh)
    limits = [coords.min(), coords.max()]

    # set view
    if hemi == 'rh':
        if view == 'lateral':
            elev, azim = 0, 0
        elif view == 'medial':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal or ventral')
    elif hemi == 'lh':
        if view == 'medial':
            elev, azim = 0, 0
        elif view == 'lateral':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal or ventral')
    else:
        raise ValueError('hemi must be one of rh or lh')

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if cmap is given as string, translate to matplotlib cmap
    if isinstance(cmap, _basestring):
        cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # plot mesh without data
    plot_surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = load_surf_data(bg_map)
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_faces[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            face_colors = plt.cm.gray_r(bg_faces)

        # modify alpha values in face colors
        face_colors[:, 3] = alpha*face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = load_surf_data(stat_map)
            if stat_map_data.shape[0] != coords.shape[0]:
                raise ValueError('The stat_map does not have the same number '
                                 'of vertices as the mesh. For plotting of '
                                 'rois or labels use plot_roi_surf instead')
            stat_map_values = np.mean(stat_map_data[faces], axis=1)

            cbar_vmin, cbar_vmax, vmin, vmax = \
                _get_plot_surf_params(stat_map_values, vmax,
                                      symmetric_cbar, kwargs)

            if threshold is not None:
                kept_indices = np.where(abs(stat_map_values) >= threshold)[0]
                stat_map_values = stat_map_values - vmin
                stat_map_values = stat_map_values / (vmax-vmin)
                face_colors[kept_indices] = cmap(stat_map_values[kept_indices])
            else:
                stat_map_values = stat_map_values - vmin
                stat_map_values = stat_map_values / (vmax-vmin)
                face_colors = cmap(stat_map_values)

        plot_mesh.set_facecolors(face_colors)

    # save figure if output file is given
    if output_file is not None:
        plt.savefig(output_file)

    return plot_mesh
