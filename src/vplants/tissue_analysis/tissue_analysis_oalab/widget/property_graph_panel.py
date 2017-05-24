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

from openalea.container import PropertyGraph, TemporalPropertyGraph

try:
    from vplants.tissue_analysis.tissue_analysis_oalab.property_graph_to_triangular_mesh import property_graph_to_triangular_mesh
except:
    print "Openalea.Cellcomplex must be installed to use PropertyGraphControls!"
    raise

try:
    from vplants.tissue_analysis.tissue_analysis_oalab.property_graph_to_dataframe import property_graph_to_dataframe
except:
    print "You should have pandas installed to use data visualization features! (pip install pandas)"
    dataframe = False
    pass
else:
    dataframe = True



import numpy as np

from tissuelab.gui.vtkviewer.vtkworldviewer import setdefault, world_kwargs


# element_names = dict(zip(range(4),['vertices','edges','faces','cells']))

# cst_proba = dict(step=0.01, min=0, max=1)
cst_offset = dict(step=0.1,min=0,max=2)
cst_edges = dict(enum=['spatio-temporal','spatial','temporal'])
cst_extent_range = dict(step=1, min=-1, max=101)

attribute_definition = {}

attribute_definition['property_graph'] = {}
# attribute_definition['tissue_analysis']["display_image"] = dict(value=False,interface="IBool",constraints={},label="Display Image") 
attribute_definition['property_graph']['vertex_property_name'] = dict(value="",interface="IEnumStr",constraints=dict(enum=[""]),label="Vertex Property")  
attribute_definition['property_graph']["display_vertices"] = dict(value=False,interface="IBool",constraints={},label="Display Vertices")  
attribute_definition['property_graph']["vertex_dataframe"] = dict(value=False,interface="IBool",constraints={},label="Vertices Data")  

attribute_definition['property_graph']['edge_property_name'] = dict(value="",interface="IEnumStr",constraints=dict(enum=[""]),label="Edge Property")  
attribute_definition['property_graph']["display_edges"] = dict(value=False,interface="IBool",constraints={},label="Display Edges")  
attribute_definition['property_graph']["edge_dataframe"] = dict(value=False,interface="IBool",constraints={},label="Edges Data")  


for axis in ['x', 'y', 'z']:
    label = u"Move " + axis + " slice"
    attribute_definition['property_graph'][axis + "_slice"] = dict(value=(-1,101),interface="IIntRange",constraints=cst_extent_range,label=label)

attribute_definition['property_graph']["filter_property_name"] = dict(value=False,interface="IEnumStr",constraints=dict(enum=[""]),label="Filter by...")  

# attribute_definition['tissue_analysis']["cell_layer"] = dict(value="",interface="IEnumStr",constraints=cst_layers,label="Cell Layer")     
# for axis in ['x', 'y', 'z']:
#     label = u"Move " + axis + " slice"
#     attribute_definition['tissue_analysis'][axis + "_slice"] = dict(value=(-1,101),interface="IIntRange",constraints=cst_extent_range,label=label)


attribute_definition['temporal_property_graph'] = {}

attribute_definition['temporal_property_graph']['edge_type'] = dict(value='spatio-temporal',interface="IEnumStr",constraints=cst_edges,label="Edge Type") 
attribute_definition['temporal_property_graph']['time_point'] = dict(value=0,interface="IInt",constraints=dict(min=0,max=0,step=1),label="Time Point") 
attribute_definition['temporal_property_graph']['all_time_points'] = dict(value=False,interface="IBool",constraints={},label="All Time Points")  
attribute_definition['temporal_property_graph']['time_offset'] = dict(value=1,interface="IFloat",constraints=cst_offset,label="Time Offset")  

def _property_names(world_object, attr_name, property_name, **kwargs):
    element = attr_name[:-14]
    graph = world_object.data
    if element == 'vertex':
        constraints = dict(enum=[""]+list(graph.vertex_property_names()))
    elif element == 'edge':
        constraints = dict(enum=[""]+list(graph.edge_property_names()))
    if property_name in constraints['enum']:
        return dict(value=property_name, constraints=constraints)
    else:
        return dict(value="", constraints=constraints)

def _bool_property_names(world_object, attr_name, property_name, **kwargs):

    graph = world_object.data
    constraints = dict(enum=[""]+list([p for p in graph.vertex_property_names() if np.array(graph.vertex_property(p).values()).dtype==bool]))
    if property_name in constraints['enum']:
        return dict(value=property_name, constraints=constraints)
    else:
        return dict(value="", constraints=constraints)

def _time_points(world_object, attr_name, time_point, **kwargs):

    graph = world_object.data
    constraints = dict(min=0,max=graph.nb_time_points-1,step=1)
    print constraints
    if time_point < graph.nb_time_points:
        return dict(value=time_point, constraints=constraints)
    else:
        return dict(value=0, constraints=constraints)


class PropertyGraphControlPanel(QtGui.QWidget, AbstractListener):
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

        self._graph = None

        self._mesh = {}

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
            if isinstance(world[object_name].data,PropertyGraph):
                self.refresh_world_object(world[object_name])
    
    def notify(self, sender, event=None):
        signal, data = event
        if signal == 'world_changed':
            world, old_object, new_object = data
            if isinstance(new_object.data,PropertyGraph):
                self.refresh()
        elif signal == 'world_object_removed':
            world, old_object = data
            if isinstance(old_object.data,PropertyGraph):
                if world.has_key(old_object.name+"_vertices"):
                    world.remove(old_object.name+"_vertices")
                if world.has_key(old_object.name+"_edges"):
                    world.remove(old_object.name+"_edges")
                del self._mesh[old_object.name]
                self.refresh()
        elif signal == 'world_object_changed':
            world, old_object, world_object = data
            if isinstance(world_object.data,PropertyGraph):
                self.refresh_world_object(world_object)
        elif signal == 'world_object_item_changed':
            world, world_object, item, old, new = data
            if isinstance(world_object.data,PropertyGraph):
                # self.refresh_manager(world_object)
                if item == 'attribute':
                    self.update_graph_display(world_object, new)
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
        self._mesh = {}
        if self.world:
            self.world.unregister_listener(self)
            self.world = None

    def refresh_world_object(self, world_object):
        if world_object:
            dtype = 'property_graph'

            graph = world_object.data
            self._graph = graph
            kwargs = world_kwargs(world_object)

            if isinstance(graph,TemporalPropertyGraph):
                if not 'mother_cell' in graph.vertex_property_names():
                    mother_cells = dict(zip(list(graph.vertices()),[np.min(list(graph.ancestors(v))) for v in graph.vertices()]))
                    print mother_cells
                    graph.add_vertex_property('mother_cell',mother_cells)

            print "Set default attributes : ",world_object.name

            world_object.silent = True
            setdefault(world_object, dtype, 'vertex_property_name', conv=_property_names, attribute_definition=attribute_definition, **kwargs)
            setdefault(world_object, dtype, 'display_vertices', attribute_definition=attribute_definition, **kwargs)
            if dataframe:
                setdefault(world_object, dtype, 'vertex_dataframe', attribute_definition=attribute_definition, **kwargs)

            setdefault(world_object, dtype, 'edge_property_name', conv=_property_names, attribute_definition=attribute_definition, **kwargs)
            setdefault(world_object, dtype, 'display_edges', attribute_definition=attribute_definition, **kwargs)
            if dataframe:
                setdefault(world_object, dtype, 'edge_dataframe', attribute_definition=attribute_definition, **kwargs)

            if isinstance(graph,TemporalPropertyGraph):
                setdefault(world_object, 'temporal_property_graph', 'edge_type', attribute_definition=attribute_definition, **kwargs)

            for axis in ['x','y','z']:
                setdefault(world_object, dtype, axis+"_slice", attribute_definition=attribute_definition, **kwargs)

            setdefault(world_object, dtype, 'filter_property_name', conv=_bool_property_names, attribute_definition=attribute_definition, **kwargs)

            if isinstance(graph,TemporalPropertyGraph):
                setdefault(world_object, 'temporal_property_graph', 'time_point', conv=_time_points, attribute_definition=attribute_definition, **kwargs)
                setdefault(world_object, 'temporal_property_graph', 'all_time_points', attribute_definition=attribute_definition, **kwargs)
                setdefault(world_object, 'temporal_property_graph', 'time_offset', attribute_definition=attribute_definition, **kwargs)

            world_object.silent = False

            # if not self._mesh.has_key(world_object.name):
            self._mesh[world_object.name] = dict(vertex=None,edge=None)
                
            world_object.set_attribute("display_vertices",True)
            world_object.set_attribute("display_edges",True)
    

    def select_world_object(self, object_name):
        if object_name != self._current:
            self._current = object_name

    def refresh_item(self, world_object, item, old, new):
        object_name = world_object.name


    def refresh(self):
        if self.world is not None:
            self.set_world(self.world)

    def update_graph_display(self, world_object, attribute):
        if world_object:

            graph = world_object.data

            if isinstance(graph,TemporalPropertyGraph):
                temporal = True
            else:
                temporal = False

            # labels = list(graph.vertices())

            if world_object['filter_property_name'] == "":
                labels = list(graph.vertices())
            else:
                filter_property = graph.vertex_property(world_object['filter_property_name'])
                labels = [v for v in filter_property.keys() if filter_property[v]]

            labels = [v for v in labels if graph.vertex_property('barycenter').has_key(v)]

            extent = np.transpose([np.array(graph.vertex_property('barycenter').values()).min(axis=0),np.array(graph.vertex_property('barycenter').values()).max(axis=0)])
            for dim,axis in enumerate(['x', 'y', 'z']):
                dim_slice = [extent[dim,0] + s*(extent[dim,1]-extent[dim,0])/100. for s in world_object[axis+"_slice"]]
                labels = [v for v in labels if graph.vertex_property('barycenter')[v][dim]>=dim_slice[0]]
                labels = [v for v in labels if graph.vertex_property('barycenter')[v][dim]<=dim_slice[1]]

            if temporal:
                if not world_object['all_time_points']:
                    time_point = world_object['time_point']
                    labels = [v for v in labels if graph.vertex_temporal_index(v) == time_point]

            # if world_object['cell_layer'] == 'L1':
            #     labels = sia.cell_first_layer()
            # elif world_object['cell_layer'] == 'L2':
            #     labels = sia.cell_second_layer()
            # else:
            #     labels = sia.labels()

            for element, element_plural in zip(['vertex','edge'],['vertices','edges']):

                if 'display_'+element_plural in attribute['name'] or attribute['name'] in ['filter_property_name','time_point','edge_type','all_time_points','time_offset'] or '_slice' in attribute['name']:
                    if world_object['display_'+element_plural]:
                        property_name = world_object[element+'_property_name']
                        
                        if temporal:
                            edge_type = world_object['edge_type']
                            time_offset = world_object['time_offset']
                            if not world_object['all_time_points']:
                                mesh = property_graph_to_triangular_mesh(graph,element,property_name,labels,edge_type=edge_type)
                            else:
                                mesh = property_graph_to_triangular_mesh(graph,element,property_name,labels,edge_type=edge_type,time_offset=time_offset*(extent[0,1]-extent[0,0]))
                        else:
                            mesh = property_graph_to_triangular_mesh(graph,element,property_name,labels)

                        if self.world.has_key(world_object.name+"_"+element_plural):
                            kwargs = world_kwargs(self.world[world_object.name+"_"+element_plural])
                        else:
                            kwargs = {}
                            kwargs['colormap'] = ('glasbey' if element=='vertex' else 'grey') if (property_name == '') else 'jet'

                        self._mesh[world_object.name][element] = mesh

                        self.world.add(self._mesh[world_object.name][element],world_object.name+"_"+element_plural,**kwargs)
                    else:
                        self.world.remove(world_object.name+"_"+element_plural)

                elif element+'_property_name' in attribute['name']:
                    if world_object['display_'+element_plural]:
                        property_name = world_object[element+'_property_name']
                        
                        if element == 'vertex':
                            if property_name in graph.vertex_property_names():

                                graph_property = graph.vertex_property(property_name)
                                property_data = array_dict(dict(zip(labels,[graph_property[l] if graph_property.has_key(l) else np.nan for l in labels ])))
                            else:
                                property_data = array_dict(dict(zip(labels,labels)))
                            self._mesh[world_object.name][element].point_data = property_data
                        elif element == 'edge':
                            graph_edges = self._mesh[world_object.name][element].edges.keys()
                            if property_name in graph.edge_property_names():
                                graph_property = graph.edge_property(property_name)
                                property_data = array_dict(dict(zip(graph_edges,[graph_property[e] if graph_property.has_key(e) else np.nan for e in graph_edges])))
                            else:
                                property_data = array_dict({})
                            self._mesh[world_object.name][element].edge_data = property_data

                        self.world[world_object.name+"_"+element_plural].data = self._mesh[world_object.name][element]
                        if len(property_data)>1:
                            self.world[world_object.name+"_"+element_plural].set_attribute('intensity_range',(property_data.values().min(),property_data.values().max()))

                elif element+'_dataframe' in attribute['name']:
                    if world_object[element+'_dataframe']:

                        property_name = world_object[element+'_property_name']

                        df = property_graph_to_dataframe(graph,element,labels)
                        self.world.add(df,world_object.name+"_"+element+"_dataframe")
                        self.world[world_object.name+"_"+element+"_dataframe"].set_attribute('X_variable',property_name)
                        self.world[world_object.name+"_"+element+"_dataframe"].set_attribute('plot','cumulative')

                    else:
                        self.world.remove(world_object.name+"_"+element+"_dataframe")










