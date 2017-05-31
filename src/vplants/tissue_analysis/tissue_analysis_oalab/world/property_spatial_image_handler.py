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
from openalea.core.observer import AbstractListener
from openalea.oalab.widget.world import WorldModel
from vplants.tissue_analysis.property_spatial_image import PropertySpatialImage, property_spatial_image_to_triangular_mesh, property_spatial_image_to_dataframe

from tissuelab.gui.vtkviewer.vtkworldviewer import setdefault, world_kwargs


cst_extent_range = dict(step=1, min=-1, max=101)

attribute_definition = {}
attribute_definition['property_image'] = {}
attribute_definition['property_image']['property_name'] = dict(value=0,interface="IEnumStr",constraints=dict(enum=[""]),label="Property")
attribute_definition['property_image']['filter_property_name'] = dict(value=0,interface="IEnumStr",constraints=dict(enum=[""]),label="Filter Property")
attribute_definition['property_image']['filter_range'] = dict(value=(-1,101),interface="IIntRange",constraints=cst_extent_range,label="Filter Range")
attribute_definition['property_image']["display_mesh"] = dict(value=True,interface="IBool",constraints={},label="Display Mesh")
attribute_definition['property_image']["display_image"] = dict(value=False,interface="IBool",constraints={},label="Display Image")
attribute_definition['property_image']["display_data"] = dict(value=True,interface="IBool",constraints={},label="Display Data")
for axis in ['x', 'y', 'z']:
    label = u"Move " + axis + " slice"
    attribute_definition['property_image'][axis + "_slice"] = dict(value=(-1,101),interface="IIntRange",constraints=cst_extent_range,label=label)

def _property_names(world_object, attr_name, property_name, **kwargs):
    print "New property_names ! "
    property_img = world_object.data
    constraints = dict(enum=[""]+list(property_img.image_property_names()))
    print constraints
    if property_name in constraints['enum']:
        return dict(value=property_name, constraints=constraints)
    else:
        return dict(value="", constraints=constraints)


class PropertyImageHandler(AbstractListener):

    def __init__(self, parent=None, style=None):
        AbstractListener.__init__(self)

        self.world = None
        self.model = WorldModel()

        self._meshes = {}


    def initialize(self):
        from openalea.core.world.world import World
        world = World()
        self.set_world(world)

    def set_world(self, world):
        self.clear()

        self.world = world
        self.world.register_listener(self)
        self.model.set_world(world)

        for object_name in world.keys():
            if isinstance(world[object_name].data,PropertySpatialImage):
                self.refresh_world_object(world[object_name])

    def clear(self):
        if self.world:
            self.world.unregister_listener(self)
            self.world = None


    def refresh(self):
        if self.world is not None:
            self.set_world(self.world)


    def notify(self, sender, event=None):
        signal, data = event

        if signal == 'world_changed':
            world, old_object, new_object = data
            if isinstance(new_object.data,PropertySpatialImage):
                self.refresh()
        elif signal == 'world_object_removed':
            world, old_object = data
            if isinstance(old_object.data,PropertySpatialImage):
                # figure = plt.figure(self._figures[old_object.name])
                # figure.clf()
                # del self._figures[old_object.name]
                self.refresh()
        elif signal == 'world_object_changed':
            world, old_object, world_object = data
            if isinstance(world_object.data,PropertySpatialImage):
                print world_object.name," : ",world_object.kwargs
                self.refresh_world_object(world_object)
        elif signal == 'world_object_item_changed':
            world, world_object, item, old, new = data
            if isinstance(world_object.data,PropertySpatialImage):
                # self.refresh_manager(world_object)
                if item == 'attribute':
                    self.update_image_display(world_object, new['name'])

                    # if new['name'] in ['X_range','Y_range']:
                    #     self.update_plot_limits(world_object)
                    # elif new['name'] in ['title']:
                    #     self.update_figure_title(world_object)
                    # else:
                    #     self.update_dataframe_figure(world_object)
        elif signal == 'world_sync':
            self.refresh()

    def refresh_world_object(self, world_object):
        if world_object:
            dtype = 'property_image'

            property_img = world_object.data
            kwargs = world_kwargs(world_object)
            for name,value in world_object.kwargs.items():
                kwargs[name] = value

            world_object.silent = True

            setdefault(world_object, dtype, 'property_name', conv=_property_names, attribute_definition=attribute_definition, **kwargs)

            setdefault(world_object, dtype, 'display_mesh', attribute_definition=attribute_definition, **kwargs)

            for axis in ['x', 'y', 'z']:
                setdefault(world_object, dtype, axis+'_slice', attribute_definition=attribute_definition, **kwargs)

            setdefault(world_object, dtype, 'display_data', attribute_definition=attribute_definition, **kwargs)

            setdefault(world_object, dtype, 'filter_property_name', conv=_property_names, attribute_definition=attribute_definition, **kwargs)
            
            setdefault(world_object, dtype, 'filter_range', attribute_definition=attribute_definition, **kwargs)

            world_object.silent = False

            setdefault(world_object, dtype, 'display_image', attribute_definition=attribute_definition, **kwargs)

    def update_image_display(self, world_object, attribute_name):
        if world_object:
            property_img = world_object.data
            property_name = world_object['property_name']
            filter_name = world_object['filter_property_name']

            if attribute_name == 'filter_property_name':
                if filter_name in property_img.image_property_names():
                    filter_extent = [property_img.image_property(filter_name).values().min(),property_img.image_property(filter_name).values().max()]
                else:
                    filter_extent = [-1,101]
                world_object.set_attribute('filter_range',value=tuple(filter_extent),constraints=dict(min=int(filter_extent[0]),max=int(filter_extent[1])))


            labels = property_img.labels

            if 'barycenter' in property_img.image_property_names():
                extent = np.transpose([np.array(property_img.image_property('barycenter').values()).min(axis=0),np.array(property_img.image_property('barycenter').values()).max(axis=0)])
                for dim,axis in enumerate(['x', 'y', 'z']):
                    dim_slice = [extent[dim,0] + s*(extent[dim,1]-extent[dim,0])/100. for s in world_object[axis+"_slice"]]
                    labels = [v for v in labels if property_img.image_property('barycenter')[v][dim]>=dim_slice[0]]
                    labels = [v for v in labels if property_img.image_property('barycenter')[v][dim]<=dim_slice[1]]

            if filter_name in property_img.image_property_names():
                # filter_extent = [property_img.image_property(filter_name).values().min(),property_img.image_property(filter_name).values().max()]
                # filter_slice = [filter_extent[0] + s*(filter_extent[1]-filter_extent[0])/100. for s in world_object["filter_range"]]
                # labels = [v for v in labels if property_img.image_property(filter_name)[v]>=filter_slice[0]]
                # labels = [v for v in labels if property_img.image_property(filter_name)[v]<=filter_slice[1]]
                filter_range = world_object["filter_range"]
                labels = [v for v in labels if property_img.image_property(filter_name)[v]>=filter_range[0]]
                labels = [v for v in labels if property_img.image_property(filter_name)[v]<=filter_range[1]]

            if world_object['display_mesh']:
                mesh,_ = property_spatial_image_to_triangular_mesh(property_img,property_name,labels)

                if self.world.has_key(world_object.name+"_mesh"):
                    kwargs = world_kwargs(self.world[world_object.name+"_mesh"])
                    if attribute_name in ['property_name','display_mesh']:
                        if kwargs.has_key('intensity_range'):
                            kwargs.pop('intensity_range')
                else:
                    kwargs = {}
                self.world.add(mesh,world_object.name+"_mesh",**kwargs)
            else:
                if self.world.has_key(world_object.name+"_mesh"):
                    self.world.remove(world_object.name+"_mesh")

            if world_object['display_data']:
                df = property_spatial_image_to_dataframe(property_img)

                df = df[np.any([df['label']==l for l in labels],axis=0)]

                if self.world.has_key(world_object.name+"_data"):
                    kwargs = world_kwargs(self.world[world_object.name+"_data"])
                else:
                    kwargs = {}
                self.world.add(df,world_object.name+"_data",**kwargs)
            else:
                if self.world.has_key(world_object.name+"_data"):
                    self.world.remove(world_object.name+"_data")
  

            if world_object['display_image']:
                # img = property_img.create_property_image(property_name)
                img = property_img.image

                if self.world.has_key(world_object.name+"_image"):
                    kwargs = world_kwargs(self.world[world_object.name+"_image"])
                    if kwargs.has_key('intensity_range'):
                        kwargs.pop('intensity_range')
                else:
                    kwargs = {}
                    kwargs['alphamap'] = 'constant'
                    kwargs['colormap'] = 'glasbey'
                    kwargs['bg_id'] = property_img.background

                self.world.add(img,world_object.name+"_image",**kwargs)
            else:
                if self.world.has_key(world_object.name+"_image"):
                    self.world.remove(world_object.name+"_image")



