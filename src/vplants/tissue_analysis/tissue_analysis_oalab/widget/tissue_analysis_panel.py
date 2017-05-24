# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015-2016 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Guillaume Baty <guillaume.baty@inria.fr>, 
#                            Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

__revision__ = ""

# import weakref
from openalea.vpltk.qt import QtGui, QtCore
from openalea.core.observer import AbstractListener
from openalea.core.control import Control
from openalea.oalab.control.manager import ControlManagerWidget
from openalea.core.service.ipython import interpreter as get_interpreter

from openalea.oalab.service.drag_and_drop import add_drop_callback
from openalea.oalab.widget.world import WorldModel

from openalea.container import array_dict

from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis3D

try:
    from vplants.tissue_analysis.tissue_analysis_oalab.sia_to_triangular_mesh import spatial_image_analysis_to_triangular_mesh, spatial_image_analysis_property
    from vplants.tissue_analysis.tissue_analysis_oalab.sia_to_triangular_mesh import spatial_image_analysis_to_cell_triangular_meshes, composed_triangular_mesh 
except:
    print "Openalea.Cellcomplex must be installed to use TissueAnalysisControls!"
    raise


import numpy as np

from tissuelab.gui.vtkviewer.vtkworldviewer import setdefault, world_kwargs


# element_names = dict(zip(range(4),['vertices','edges','faces','cells']))

# cst_proba = dict(step=0.01, min=0, max=1)
# cst_degree = dict(step=1,min=0,max=3)
cst_properties = dict(enum=["","volume","neighborhood_size",'shape_anisotropy',"gaussian_curvature"])
cst_layers = dict(enum=['','L1','L2'])
cst_extent_range = dict(step=1, min=-1, max=101)

attribute_definition = {}
# attribute_definition['topomesh'] = {}
# for degree in xrange(4):
#     attribute_definition['topomesh']["display_"+str(degree)] = dict(value=False,interface="IBool",constraints={},label="Display "+element_names[degree])
#     attribute_definition['topomesh']["property_degree_"+str(degree)] = dict(value=degree,interface="IInt",constraints=cst_degree,label="Degree") 
#     attribute_definition['topomesh']["property_name_"+str(degree)] = dict(value="",interface="IEnumStr",constraints=dict(enum=[""]),label="Property")     
#     attribute_definition['topomesh']["coef_"+str(degree)] = dict(value=1,interface="IFloat",constraints=cst_proba,label="Coef") 
# attribute_definition['topomesh']["filename"] = dict(value="",interface="IFileStr",constraints={},label="Filename")
# attribute_definition['topomesh']["save"] = dict(value=(lambda:None),interface="IAction",constraints={},label="Save PropertyTopomesh")

attribute_definition['tissue_analysis'] = {}
# attribute_definition['tissue_analysis']["display_image"] = dict(value=False,interface="IBool",constraints={},label="Display Image") 
attribute_definition['tissue_analysis']['property_name'] = dict(value="",interface="IEnumStr",constraints=cst_properties,label="Property")  
attribute_definition['tissue_analysis']["display_mesh"] = dict(value=False,interface="IBool",constraints={},label="Display Mesh")     
attribute_definition['tissue_analysis']["cell_layer"] = dict(value="",interface="IEnumStr",constraints=cst_layers,label="Cell Layer")     
for axis in ['x', 'y', 'z']:
    label = u"Move " + axis + " slice"
    attribute_definition['tissue_analysis'][axis + "_slice"] = dict(value=(-1,101),interface="IIntRange",constraints=cst_extent_range,label=label)

class TissueAnalysisControlPanel(QtGui.QWidget, AbstractListener):
    StyleTableView = 0
    StylePanel = 1
    DEFAULT_STYLE = StylePanel
    
    def __init__(self, parent=None, style=None):
        AbstractListener.__init__(self)
        QtGui.QWidget.__init__(self, parent=parent)

        self.world = None
        self.model = WorldModel()

        if style is None:
            style = self.DEFAULT_STYLE
        self.style = style

        # self._manager = {}

        # self._cb_world_object = QtGui.QComboBox()
        # p = QtGui.QSizePolicy
        # self._cb_world_object.setSizePolicy(p(p.Expanding, p.Maximum))
        # self._cb_world_object.currentIndexChanged.connect(self._selected_object_changed)

        self._sia = None

        self._cell_meshes = {}
        self._mesh = {}
        self._mesh_matching = {}

        self._current = None
        # self._default_manager = self._create_manager()

        self.interpreter = get_interpreter()
        self.interpreter.locals['tissue_analysis_control'] = self

        self._layout = QtGui.QVBoxLayout(self)
        # self._layout.addWidget(self._cb_world_object)

        if self.style == self.StyleTableView:
            self._view = None
            # self._view = ControlManagerWidget(manager=self._default_manager)
            # self._layout.addWidget(self._view)
        elif self.style == self.StylePanel:
            self._view = None
            # self._set_manager(self._default_manager)
        else:
            raise NotImplementedError('style %s' % self.style)

    # def set_properties(self, properties):
    #     if self.style == self.StyleTableView:
    #         self._view.set_properties(properties)

    # def properties(self):
    #     if self.style == self.StyleTableView:
    #         return self._view.properties()
    #     else:
    #         return []

    def set_style(self, style):
        if style == self.style:
            return

        world = self.world
        self.clear()
        if self.style == self.StyleTableView:
            view = self._view
        elif self.style == self.StylePanel:
            if self._view and self._view():
                view = self._view()
            else:
                return

        # Remove old view
        view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._layout.removeWidget(view)
        view.close()
        del view
        self._view = None

        self.style = style
        # if style == self.StyleTableView:
            # self._view = ControlManagerWidget(manager=self._default_manager)
            # self._layout.addWidget(self._view)

        self.set_world(world)

    # def __getitem__(self, key):
    #     return self._manager[self._current].control(name=key)

    def initialize(self):
        from openalea.core.world.world import World
        from openalea.core.service.ipython import interpreter
        world = World()
        world.update_namespace(interpreter())
        self.set_world(world)


    def set_world(self, world):
        self.clear()

        self.world = world
        self.world.register_listener(self)

        if self.style == self.StyleTableView:
            self.model.set_world(world)

        for object_name in world.keys():
            if isinstance(world[object_name].data,SpatialImageAnalysis3D):
                self.refresh_world_object(world[object_name])
    
    def notify(self, sender, event=None):
        signal, data = event
        if signal == 'world_changed':
            world, old_object, new_object = data
            if isinstance(new_object.data,SpatialImageAnalysis3D):
                self.refresh()
        elif signal == 'world_object_removed':
            world, old_object = data
            if isinstance(old_object.data,SpatialImageAnalysis3D):
                if world.has_key(old_object.name+"_mesh"):
                    world.remove(old_object.name+"_mesh")
                del self._cell_meshes[old_object.name]
                self.refresh()
        elif signal == 'world_object_changed':
            world, old_object, world_object = data
            if isinstance(world_object.data,SpatialImageAnalysis3D):
                self.refresh_world_object(world_object)
        elif signal == 'world_object_item_changed':
            world, world_object, item, old, new = data
            if isinstance(world_object.data,SpatialImageAnalysis3D):
                # self.refresh_manager(world_object)
                if item == 'attribute':
                    self.update_tissue_analysis_display(world_object, new)
        elif signal == 'world_sync':
            self.refresh()

    # def clear_managers(self):
    #     self._current = None
    #     self._cb_world_object.clear()
    #     for name, manager in self._manager.items():
    #         manager.clear_followers()
    #         del self._manager[name]
    #     self._set_manager(self._default_manager)

    def clear(self):
        # self.clear_managers()
        self._cell_meshes = {}
        self._mesh = {}
        self._mesh_matching = {}

        if self.world:
            self.world.unregister_listener(self)
            self.world = None

    def refresh_world_object(self, world_object):
        if world_object:
            dtype = 'tissue_analysis'

            self._sia = world_object.data
            kwargs = world_kwargs(world_object)

            print "Set default attributes : ",world_object.name

            world_object.silent = True
            setdefault(world_object, dtype, 'property_name', attribute_definition=attribute_definition, **kwargs)
            setdefault(world_object, dtype, 'cell_layer', attribute_definition=attribute_definition, **kwargs)
            for axis in ['x', 'y', 'z']:
                setdefault(world_object, dtype, axis+'_slice', attribute_definition=attribute_definition, **kwargs)
            world_object.silent = False
            setdefault(world_object, dtype, 'display_mesh', attribute_definition=attribute_definition, **kwargs)
            
            if not self._mesh.has_key(world_object.name):
                sia = world_object.data
                self._cell_meshes[world_object.name] = spatial_image_analysis_to_cell_triangular_meshes(sia)
                self._mesh[world_object.name] = None
                self._mesh_matching[world_object.name] = None
    

    def select_world_object(self, object_name):
        if object_name != self._current:
            self._current = object_name


    def refresh_item(self, world_object, item, old, new):
        object_name = world_object.name


    def refresh(self):
        if self.world is not None:
            self.set_world(self.world)

    def update_tissue_analysis_display(self, world_object, attribute):
        from time import time
        start_time = time()
        print "--> Updating Analysis display"

        if world_object:
            sia = world_object.data

            label_start_time = time()
            if world_object['cell_layer'] == 'L1':
                labels = sia.cell_first_layer()
            elif world_object['cell_layer'] == 'L2':
                labels = sia.cell_second_layer()
            else:
                labels = sia.labels()
            label_end_time = time()
            print "  --> Updating displayed labels   [",label_end_time - label_start_time,"s]"

            if 'display_mesh' in attribute['name'] or 'cell_layer' in attribute['name']:
                if world_object['display_mesh']:
                    sia = world_object.data
                    property_name = world_object['property_name']


                    # if self._mesh[world_object.name] is None:
                    if True:
                        cell_meshes = self._cell_meshes[world_object.name]
                        label_meshes = dict(zip(labels,[cell_meshes[l] for l in labels if cell_meshes.has_key(l)]))
                        mesh, matching = composed_triangular_mesh(label_meshes)
                        #spatial_image_analysis_to_triangular_mesh(sia,property_name,labels)
                        self._mesh[world_object.name] = mesh
                        self._mesh_matching[world_object.name] = matching
                        property_data = spatial_image_analysis_property(sia,property_name,labels)                
                        self._mesh[world_object.name].triangle_data = dict(zip(matching.keys(),property_data.values(matching.values())))

                    if self.world.has_key(world_object.name+"_mesh"):
                        kwargs = world_kwargs(self.world[world_object.name+"_mesh"])
                        # if not 'coef_' in attribute['name']:
                        #     if kwargs.has_key('intensity_range'):
                        #         kwargs.pop('intensity_range')
                    else:
                        kwargs = {}
                        kwargs['colormap'] = 'glasbey' if (property_name == '') else 'jet'
                        # kwargs['position'] = world_object['position']

                    self.world.add(self._mesh[world_object.name],world_object.name+"_mesh",**kwargs)
                else:
                    self.world.remove(world_object.name+"_mesh")

            elif 'property_name' in attribute['name']:
                if world_object['display_mesh']:
                    property_name = world_object['property_name']
                    matching = self._mesh_matching[world_object.name]

                    # labels = sia.labels()
                    property_data = spatial_image_analysis_property(sia,property_name,labels)                
                    self._mesh[world_object.name].triangle_data = dict(zip(matching.keys(),property_data.values(matching.values())))

                    self.world[world_object.name+"_mesh"].data = self._mesh[world_object.name]
                    if len(property_data)>1:
                        self.world[world_object.name+"_mesh"].set_attribute('intensity_range',(property_data.values().min(),property_data.values().max()))

        end_time = time()
        print "<-- Updating Analysis display [",end_time-start_time,"s]"








