# -*- python -*-
#
#       OpenAlea.TissueAnalysis
#
#       Copyright 2006 - 2016 INRIA - CIRAD - INRA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
#                       Guillaume CERUTTI <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
################################################################################

__license__ = "Cecill-C"

import scipy.ndimage as nd, numpy as np
from openalea.cellcomplex.property_topomesh.utils.image_tools import image_to_vtk_cell_polydata, vtk_polydata_to_cell_triangular_meshes
from openalea.cellcomplex.triangular_mesh import TriangularMesh

from openalea.container import array_dict

from vplants.tissue_analysis.spatial_image_analysis import fractional_anisotropy

from openalea.image.spatial_image import SpatialImage
from vplants.morpheme.vt_exec.trsf import apply_trsf, create_trsf
from vplants.morpheme.vt_exec.trsf import BalTransformation

from copy import copy, deepcopy
from time import time

def composed_triangular_mesh(triangular_mesh_dict):
    start_time = time()
    print "--> Composing triangular mesh..."

    mesh = TriangularMesh()

    triangle_cell_matching = {}

    mesh_points = np.concatenate([triangular_mesh_dict[c].points.keys() for c in triangular_mesh_dict.keys()])
    mesh_point_positions = np.concatenate([triangular_mesh_dict[c].points.values() for c in triangular_mesh_dict.keys()])
    mesh.points = dict(zip(mesh_points,mesh_point_positions))

    mesh_triangles = np.concatenate([triangular_mesh_dict[c].triangles.values() for c in triangular_mesh_dict.keys()])
    mesh.triangles = dict(zip(np.arange(len(mesh_triangles)),mesh_triangles))

    mesh_cells = np.concatenate([c*np.ones_like(triangular_mesh_dict[c].triangles.keys()) for c in triangular_mesh_dict.keys()])
    triangle_cell_matching = dict(zip(np.arange(len(mesh_triangles)),mesh_cells))


    # for c in triangular_mesh_dict.keys():
    #     cell_start_time = time()

    #     cell_mesh = triangular_mesh_dict[c]
    #     # mesh_point_max_id = np.max(mesh.points.keys()) if len(mesh.points)>0 else 0
    #     mesh.points.update(cell_mesh.points)

    #     if len(cell_mesh.triangles)>0:
    #         mesh_triangle_max_id = np.max(mesh.triangles.keys()) if len(mesh.triangles)>0 else 0
    #         mesh.triangles.update(dict(zip(list(np.array(cell_mesh.triangles.keys())+mesh_triangle_max_id),cell_mesh.triangles.values())))
    #         triangle_cell_matching.update(dict(zip(list(np.array(cell_mesh.triangles.keys())+mesh_triangle_max_id),[c for f in cell_mesh.triangles]))) 

    #     cell_end_time = time()
    #     print "  --> Adding cell ",c," (",len(cell_mesh.triangles)," triangles )    [",cell_end_time-cell_start_time,"s]"

    end_time = time()
    print "<-- Composing triangular mesh     [",end_time-start_time,"]"
    return mesh, triangle_cell_matching


def img_extent(sp_img):
    extent = []
    for ind in range(0,len(sp_img.shape)):
        extent.append(sp_img.shape[ind]*sp_img.voxelsize[ind])
    return extent


def img_resize(sp_img, sub_factor=2, option=None):
    """
    Down interpolation of image 'sp_img' by a factor 'sub_factor'. 
    For intensity image use 'option="gray"', for segmented images use 'option="label"'.
    """
    vx = sp_img.voxelsize
    poss_opt = ['gray', 'label']
    if option is None:
        option='label'
    else:
        if option not in poss_opt:
            option='label'
    
    extent = img_extent(sp_img)
    tmp_voxelsize = np.array(sp_img.voxelsize) *sub_factor
    print tmp_voxelsize
    new_shape, new_voxelsize = [], []
    for ind in range(0,len(sp_img.shape)):
        new_shape.append(int(np.ceil(extent[ind]/tmp_voxelsize[ind])))
        new_voxelsize.append(extent[ind]/new_shape[ind])
    identity_trsf = create_trsf(param_str_2='-identity', trsf_type=BalTransformation.RIGID_3D, trsf_unit=BalTransformation.REAL_UNIT)
    template_img = np.zeros((new_shape[0],new_shape[1],new_shape[2]), dtype=sp_img.dtype)
    template_img = SpatialImage(template_img, voxelsize=new_voxelsize)
    if option=='gray':
        param_str_2 = '-interpolation linear'
    elif option=='label':
        param_str_2 = '-interpolation nearest'
    out_img = apply_trsf(sp_img, identity_trsf,template_img=template_img, param_str_2=param_str_2)

    return out_img


def spatial_image_analysis_property(sia, property_name, labels=None):
    
    if property_name == 'volume':
        property_data = sia.volume(labels)
    elif property_name == 'neighborhood_size':
        property_data = sia.neighbors_number(labels)
    elif property_name == 'shape_anisotropy':
        inertia_axes_vectors, inertia_axes_values = sia.inertia_axis(labels)
        property_data = [fractional_anisotropy(inertia_axes_values[l]) for l in labels]
    elif property_name == 'gaussian_curvature':
        property_data = sia.gaussian_curvature_CGAL(labels)
    else:
        property_data = dict(zip(labels,labels))

    if isinstance(property_data,np.ndarray) or isinstance(property_data,list):
        property_data = array_dict(property_data, keys=labels)
    elif isinstance(property_data,dict):
        property_data = array_dict(property_data) 

    return property_data


def spatial_image_analysis_to_cell_triangular_meshes(input_sia, labels=None):

    sia = deepcopy(input_sia)
    background = sia.background()

    img_labels = sia.labels()
    if labels is not None:
        labels_to_remove = set(img_labels) - set(list(labels))
        sia.remove_labels_from_image(labels_to_remove, erase_value=background)

    img_labels = sia.labels()
    segmented_img = img_resize(deepcopy(sia.image), sub_factor=6)

    cell_triangular_meshes = vtk_polydata_to_cell_triangular_meshes(image_to_vtk_cell_polydata(segmented_img,coef=0.98,mesh_fineness=0.8,smooth_factor=1.0))

    return cell_triangular_meshes


def spatial_image_analysis_to_triangular_mesh(input_sia, property_name=None, labels=None):

    sia = deepcopy(input_sia)

    cell_triangular_meshes = spatial_image_analysis_to_cell_triangular_meshes(sia, labels)
    img_labels = cell_triangular_meshes.keys()
    mesh, matching = composed_triangular_mesh(cell_triangular_meshes)
    property_data = spatial_image_analysis_property(sia,property_name,img_labels)
    mesh.triangle_data = dict(zip(matching.keys(),property_data.values(matching.values())))

    return mesh, matching






