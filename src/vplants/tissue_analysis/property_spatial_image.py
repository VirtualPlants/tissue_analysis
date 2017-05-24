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

from temporal_graph_from_image import SpatialImage, graph_from_image
from openalea.container import array_dict
from openalea.cellcomplex.triangular_mesh import TriangularMesh

from copy import deepcopy

class PropertySpatialImage(object):

    def __init__(self, image=None, image_graph=None, background=1, **kwargs):
        
        self._properties = {}

        self.background = background

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
        self._image = SpatialImage(image, voxelsize=image.voxelsize, **kwargs) if image is not None else None
        
        if self._image is not None:
            if self._image_graph is None:
                self.set_image_graph(graph_from_image(self._image, background=self.background, spatio_temporal_properties=['barycenter'], ignore_cells_at_stack_margins=True))
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


    def compute_image_property(self, property_name):
        try:
            assert property_name in ['barycenter','volume','neighborhood_size','layer']
        except:
            print "Property \""+property_name+"\" can not be computed on image"
        else:
            if self._image is not None:
                if property_name in ['barycenter','volume']:
                    graph = graph_from_image(self._image,background=self.background,spatio_temporal_properties=[property_name],ignore_cells_at_stack_margins=True)
                    property_dict = graph.vertex_property(property_name)
                elif property_name == 'neighborhood_size':
                    neighbors = [self.image_graph.neighbors(l) for l in self.labels]
                    property_dict = dict(zip(self.labels,map(len,neighbors)))
                elif property_name == 'layer':
                    graph = graph_from_image(self._image,background=self.background,spatio_temporal_properties=['L1'],ignore_cells_at_stack_margins=True)
                    first_layer = graph.vertex_property('L1')
                    second_layer_cells = [v for v in graph.vertices() if np.any([first_layer[n] for n in graph.neighbors(v)]) and not first_layer[v]]
                    second_layer = dict(zip(list(graph.vertices()),[v in second_layer_cells for v in graph.vertices()]))
                    property_dict = dict(zip(self.labels,[1 if first_layer[l] else 2 if second_layer[l] else 0 for l in self.labels]))

                
                property_data = [property_dict[l] for l in self.labels]
                self.update_image_property(property_name,property_data)

    def compute_image_properties(self, property_names=['barycenter','volume','neighborhood_size','layer']):
        for property_name in property_names:
            self.compute_image_property(property_name)

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

    def _to_dataframe(self):
        import pandas as pd
        
        if self._image is not None:
            image_df = pd.DataFrame()
            image_df['label'] = self._labels
            for property_name in self.image_property_names():
                property_data = self.image_property(property_name).values(image_df['label'])
                if property_name in ['barycenter']:
                    for i,dim in enumerate(['x','y','z']):
                        image_df[property_name+"_"+dim] = property_data[:,i]
                else:
                    image_df[property_name] = property_data
            image_df.set_index('label')
            return image_df

    def compute_cell_meshes(self):
        from openalea.cellcomplex.property_topomesh.utils.image_tools import image_to_vtk_cell_polydata, vtk_polydata_to_cell_triangular_meshes, img_resize
        segmented_img = img_resize(deepcopy(self.image), sub_factor=4)
        self._cell_meshes = vtk_polydata_to_cell_triangular_meshes(image_to_vtk_cell_polydata(segmented_img,coef=0.98,mesh_fineness=0.75,smooth_factor=1.1))

    @property
    def cell_meshes(self):
        if self._cell_meshes is None:
            self.compute_cell_meshes()
        return self._cell_meshes


def property_spatial_image_to_triangular_mesh(image, property_name=None, labels=None):
    from openalea.cellcomplex.property_topomesh.utils.image_tools import composed_triangular_mesh

    cell_triangular_meshes = image.cell_meshes
    img_labels = cell_triangular_meshes.keys()
    if labels is None:
        labels = image.labels

    mesh, matching = composed_triangular_mesh(dict([(c,cell_triangular_meshes[c]) for c in labels if c in cell_triangular_meshes.keys()]))

    if property_name in image.image_property_names():
        property_dict = image.image_property(property_name)
    else:
        property_dict = array_dict(labels,keys=labels)
    
    mesh.triangle_data = dict(zip(matching.keys(),property_dict.values(matching.values())))

    return mesh, matching




                


