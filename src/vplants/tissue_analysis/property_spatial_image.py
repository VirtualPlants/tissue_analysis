# -*- coding: utf-8 -*-
# -*- python -*-
#
#       VPlants.tissue_analysis
#
#       Copyright 2006 - 2017 INRIA - CIRAD - INRA
#
#       File author(s): Eric MOSCARDI <eric.moscardi@sophia.inria.fr>
#                       Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
#                       Frederic BOUDON <frederic.boudon@cirad.fr>
#                       Guillaume CERUTTI <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
################################################################################

import numpy as np 
import scipy.ndimage as nd
from scipy.cluster.vq import vq

from temporal_graph_from_image import SpatialImage, graph_from_image
from openalea.container import array_dict
from openalea.cellcomplex.triangular_mesh import TriangularMesh

from copy import deepcopy

class PropertySpatialImage(object):

    def __init__(self, image=None, image_graph=None, background=1, ignore_cells_at_stack_margins=False, min_contact_area=None, **kwargs):
        
        self._properties = {}

        self.background = background
        self.ignore_cells_at_stack_margins = ignore_cells_at_stack_margins
        self.min_contact_area = min_contact_area

        self._image_graph = None
        self._labels = None
        self.set_image_graph(image_graph)

        self._image = None
        self.set_image(image, **kwargs)

        self._cell_meshes = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image, **kwargs):
        self.set_image(image, **kwargs)

    def set_image(self, image, **kwargs):
        voxelsize = image.voxelsize if hasattr(image,'voxelsize') else None
        self._image = SpatialImage(image, voxelsize=voxelsize, **kwargs) if image is not None else None
        
        if self._image is not None:
            if self._image_graph is None:
                self.set_image_graph(graph_from_image(self._image, background=self.background, spatio_temporal_properties=['barycenter'], ignore_cells_at_stack_margins=self.ignore_cells_at_stack_margins, min_contact_area=self.min_contact_area))
        else:
            self._image_graph = None
            self._labels = None


    @property
    def image_graph(self):
        return self._image_graph

    @image_graph.setter
    def image_graph(self, image_graph):
        self.set_image_graph(image_graph)

    def set_image_graph(self, image_graph):
        self._image_graph = image_graph
        if self._image_graph is not None:
            self._labels = np.sort(list(self._image_graph.vertices()))
            for property_name in self._image_graph.vertex_property_names():
                property_dict = self._image_graph.vertex_property(property_name)
                property_data = [property_dict[l] for l in self._labels]
                self.update_image_property(property_name,property_data)

    
    @property
    def labels(self):
        return self._labels

    def has_image_property(self, property_name, is_computed=True):
        if is_computed:
            return (property_name in self._properties.keys()) and np.all(self._properties[property_name].keys() == self.labels)
        else:
            return (property_name in self._properties.keys())

    def update_image_property(self, property_name, property_data, erase_property=False):
        if isinstance(property_data,list) or isinstance(property_data,np.ndarray):
            assert len(property_data) == len(self._labels)
            property_keys = self._labels
        elif isinstance(property_data,dict) or isinstance(property_data,array_dict):
            property_keys = np.sort(property_data.keys())
            property_data = [property_data[l] for l in property_keys]

        if property_name in self._properties.keys():
            if erase_property:
                self._properties[property_name] = array_dict(property_data,keys=property_keys)
            else:
                for l,v in zip(property_keys,property_data):
                    self._properties[property_name][l] = v
        else:
            print "Creating property ",property_name," on image"
            self._properties[property_name] = array_dict(property_data,keys=property_keys)


    def image_property(self, property_name):
        if property_name in self.image_property_names():
            return self._properties[property_name]

    def image_property_names(self):
        return np.sort(self._properties.keys())


    def compute_image_property(self, property_name, min_contact_area=None, sub_factor=8.):
        """
        """
        computable_properties = ['barycenter','volume','neighborhood_size','layer','mean_curvature','gaussian_curvature']
        try:
            assert property_name in computable_properties
        except:
            print "Property \""+property_name+"\" can not be computed on image"
            print "Try with one of the following :"
            for p in computable_properties:
                print "  * "+p
        else:
            if self._image is not None:
                if property_name in ['barycenter','volume']:
                    graph = graph_from_image(self._image,background=self.background,spatio_temporal_properties=[property_name],ignore_cells_at_stack_margins=self.ignore_cells_at_stack_margins)
                    property_dict = graph.vertex_property(property_name)
                elif property_name == 'neighborhood_size':
                    neighbors = [self.image_graph.neighbors(l) for l in self.labels]
                    property_dict = dict(zip(self.labels,map(len,neighbors)))
                elif property_name == 'layer':
                    if min_contact_area is None:
                        min_contact_area = self.min_contact_area
                    graph = graph_from_image(self._image,background=self.background,spatio_temporal_properties=['L1'],ignore_cells_at_stack_margins=self.ignore_cells_at_stack_margins, min_contact_area=min_contact_area)
                    first_layer = graph.vertex_property('L1')
                    second_layer_cells = [v for v in graph.vertices() if np.any([first_layer[n] for n in graph.neighbors(v)]) and not first_layer[v]]
                    second_layer = dict(zip(list(graph.vertices()),[v in second_layer_cells for v in graph.vertices()]))
                    property_dict = dict(zip(self.labels,[1 if first_layer[l] else 2 if second_layer[l] else 3 for l in self.labels]))
                elif property_name in ['mean_curvature','gaussian_curvature']:
                    if not self.has_image_property('barycenter'):
                        self.compute_image_property('barycenter')
                    if not self.has_image_property('layer'):
                        print "--> Computing layer property"
                        self.compute_image_property('layer')

                    cell_centers = self.image_property('barycenter')
                    L1_cells = self.labels[self.image_property('layer').values()==1]
                    
                    from openalea.cellcomplex.property_topomesh.utils.implicit_surfaces import implicit_surface_topomesh
                    from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
                    from openalea.cellcomplex.property_topomesh.property_topomesh_optimization import property_topomesh_vertices_deformation, topomesh_triangle_split

                    sub_binary_image = (self._image!=self.background).astype(float)[::sub_factor,::sub_factor,::sub_factor]
                    surface_topomesh = implicit_surface_topomesh(sub_binary_image,np.array(sub_binary_image.shape),sub_factor*np.array(self._image.voxelsize),center=False)
                    property_topomesh_vertices_deformation(surface_topomesh,iterations=10)

                    compute_topomesh_property(surface_topomesh,'barycenter',2)
                    compute_topomesh_property(surface_topomesh,'normal',2,normal_method='orientation')

                    compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',adjacency_sigma=2,neighborhood=5)
                    compute_topomesh_property(surface_topomesh,'mean_curvature',2)
                    compute_topomesh_vertex_property_from_faces(surface_topomesh,property_name,adjacency_sigma=2,neighborhood=5)

                    surface_cells = L1_cells[vq(surface_topomesh.wisp_property('barycenter',0).values(),cell_centers.values(L1_cells))[0]]
                    surface_topomesh.update_wisp_property('label',0,array_dict(surface_cells,list(surface_topomesh.wisps(0))))

                    L1_cell_property = nd.sum(surface_topomesh.wisp_property(property_name,0).values(),surface_cells,index=L1_cells)/nd.sum(np.ones_like(surface_cells),surface_cells,index=L1_cells)
                    L1_cell_property = array_dict(L1_cell_property,L1_cells)    
                    property_dict = array_dict([L1_cell_property[l] if (l in L1_cells) and (not np.isnan(L1_cell_property[l])) else 0 for l in self.labels],self.labels)
                
                property_data = [property_dict[l] for l in self.labels]
                self.update_image_property(property_name,property_data)

    def compute_default_image_properties(self):
        default_properties = ['barycenter','volume','neighborhood_size','layer']
        for property_name in default_properties:
            self.compute_image_property(property_name)

    def compute_image_property_from_function(self, property_name, property_function):
        """
        property_function : a function taking as an input a binary image and computing
        a specific cell property.
        """
        from time import time
        start_time = time()
        print "--> Computing "+property_name+" property"
        self._properties[property_name] = array_dict()
        for l in self.labels:
            label_start_time = time()
            binary_img = (self.image == l).astype(int)
            self._properties[property_name][l] = property_function(binary_img)
            print "  --> Computing label "+str(l)+" "+property_name+" [",time()-label_start_time," s]"
        print "<-- Computing "+property_name+" property [",time()-start_time," s]"

    def create_property_image(self, property_name=None, dtype=np.uint16):
        if property_name not in self.image_property_names():
            self.compute_image_property(property_name)
            if property_name not in self.image_property_names():
                property_name = None
        if property_name is None:
            return self.image
        else:
            property_dict = deepcopy(self.image_property(property_name))
            for l in np.unique(self.image):
                if not l in property_dict.keys():
                    property_dict[l] = self.background
            property_dict[self.background] = self.background
            property_image = property_dict.values(self.image)
            return SpatialImage(property_image.astype(dtype), voxelsize=self.image.voxelsize)

    def compute_cell_meshes(self, sub_factor=4):
        from openalea.cellcomplex.property_topomesh.utils.image_tools import image_to_vtk_cell_polydata, vtk_polydata_to_cell_triangular_meshes, img_resize
        # from timagetk.components import SpatialImage as TissueImage
        # segmented_img = img_resize(TissueImage(deepcopy(self.image),voxelsize=self.image.voxelsize), sub_factor=3)
        segmented_img = SpatialImage(deepcopy(self.image)[::sub_factor,::sub_factor,::sub_factor],voxelsize=tuple([s*sub_factor for s in self.image.voxelsize]))
        self._cell_meshes = vtk_polydata_to_cell_triangular_meshes(image_to_vtk_cell_polydata(segmented_img,coef=0.99,mesh_fineness=1,smooth_factor=1.5))

    @property
    def cell_meshes(self):
        if self._cell_meshes is None:
            self.compute_cell_meshes()
        return self._cell_meshes


def property_spatial_image_to_dataframe(image):
    import pandas as pd
    
    if image._image is not None:
        image_df = pd.DataFrame()
        image_df['label'] = image._labels
        for property_name in image.image_property_names():
            property_data = image.image_property(property_name).values(image_df['label'])
            if property_name in ['barycenter']:
                for i,dim in enumerate(['x','y','z']):
                    image_df[property_name+"_"+dim] = property_data[:,i]
            else:
                image_df[property_name] = property_data
        image_df.set_index('label')
        return image_df

def property_spatial_image_to_triangular_mesh(image, property_name=None, labels=None, coef=1):
    from openalea.cellcomplex.property_topomesh.utils.image_tools import composed_triangular_mesh

    cell_triangular_meshes = deepcopy(image.cell_meshes)
    img_labels = cell_triangular_meshes.keys()
    if labels is None:
        labels = image.labels

    if coef != 1:
        for l in labels:
            cell_center = np.mean(cell_triangular_meshes[l].points.values(),axis=0)
            points = cell_center + coef*(cell_triangular_meshes[l].points.values()-cell_center)
            cell_triangular_meshes[l].points = array_dict(points,cell_triangular_meshes[l].points.keys())

    mesh, matching = composed_triangular_mesh(dict([(c,cell_triangular_meshes[c]) for c in labels if c in cell_triangular_meshes.keys()]))

    if property_name in image.image_property_names():
        property_dict = image.image_property(property_name)
    else:
        property_dict = array_dict(labels,keys=labels)
    
    mesh.triangle_data = dict(zip(matching.keys(),property_dict.values(matching.values())))

    return mesh, matching




                


