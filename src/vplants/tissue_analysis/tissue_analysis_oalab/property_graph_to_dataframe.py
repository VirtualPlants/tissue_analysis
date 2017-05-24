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

import pandas as pd

def property_graph_to_dataframe(graph, element='vertex', labels=None):

    graph_labels = list(graph.vertices())
    if labels is not None:
        labels = list(set(graph_labels) & set(list(labels)))
    else:
        labels = graph_labels

    dataframe = pd.DataFrame()
    if element == 'vertex':
        dataframe['id'] = np.array(list(labels))



        for property_name in graph.vertex_property_names():
            if np.array(graph.vertex_property(property_name).values()[0]).ndim == 0:
                print "  --> Adding column ",property_name
                dataframe[property_name] = np.array([graph.vertex_property(property_name)[v] for v in labels])
            elif property_name == 'barycenter':
                for i, axis in enumerate(['x','y','z']):
                    dataframe[property_name+"_"+axis] = np.array([graph.vertex_property(property_name)[v][i] for v in labels])

    elif element == 'edge':
        graph_edges = [e for e in graph.edges() if np.all([v in labels for v in graph.edge_vertices(e)])]

        dataframe['id'] = np.array(list(graph_edges))

        for property_name in graph.edge_property_names():
            if np.array(graph.edge_property(property_name).values()[0]).ndim == 0:
                print "  --> Adding column ",property_name
                dataframe[property_name] = np.array([graph.edge_property(property_name)[e] for e in graph_edges])


    dataframe = dataframe.set_index('id')
    dataframe.index.name = None

    return dataframe