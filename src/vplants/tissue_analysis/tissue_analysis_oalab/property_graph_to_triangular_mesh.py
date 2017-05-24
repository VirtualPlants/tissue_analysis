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

from openalea.cellcomplex.triangular_mesh import TriangularMesh
from openalea.container import array_dict, PropertyGraph, TemporalPropertyGraph


from copy import copy, deepcopy
from time import time

def property_graph_to_triangular_mesh(graph, element='vertex', property_name=None, labels=None, edge_type='spatio-temporal',time_offset=0.):

    try:
        assert 'barycenter' in graph.vertex_property_names()
    except:
        raise KeyError("Property 'barycenter' should be defined to visualize your PropertyGraph!")

    if isinstance(graph,TemporalPropertyGraph):
        temporal = True
    else:
        temporal = False

    mesh = TriangularMesh()

    graph_labels = list(graph.vertices())
    if labels is not None:
        labels = list(set(graph_labels) & set(list(labels)))
    else:
        labels = graph_labels

    if temporal:
        mesh.points = dict(zip(labels,[graph.vertex_property('barycenter')[l] + time_offset*np.array([graph.vertex_temporal_index(l),0,0]) for l in labels]))
    else:
        mesh.points = dict(zip(labels,[graph.vertex_property('barycenter')[l] for l in labels]))

    if element == 'vertex':
        if property_name in graph.vertex_property_names():
            graph_property = graph.vertex_property(property_name)
            mesh.point_data = dict(zip(labels,[graph_property[l] if graph_property.has_key(l) else np.nan for l in labels ]))
        else:
            mesh.point_data = dict(zip(labels,labels))

    elif element == 'edge':
        if temporal:
            if edge_type == 'spatio-temporal':
                graph_edges = [e for e in graph.edges() if np.all([v in labels for v in graph.edge_vertices(e)])]
            elif edge_type == 'spatial':
                graph_edges = [e for e in graph.edges() if (graph.edge_property('edge_type')[e]=='s') and np.all([v in labels for v in graph.edge_vertices(e)])]
            elif edge_type == 'temporal':
                graph_edges = [e for e in graph.edges() if (graph.edge_property('edge_type')[e]=='t') and np.all([v in labels for v in graph.edge_vertices(e)])]


        else:
            graph_edges = [e for e in graph.edges() if np.all([v in labels for v in graph.edge_vertices(e)])]
        mesh.edges = dict(zip(graph_edges,[graph.edge_vertices(e) for e in graph_edges]))
        if property_name in graph.edge_property_names():
            graph_property = graph.edge_property(property_name)
            mesh.edge_data = dict(zip(graph_edges,[graph_property[e] if graph_property.has_key(e) else np.nan for e in graph_edges]))
        else:
            mesh.edge_data = {}

    return mesh

