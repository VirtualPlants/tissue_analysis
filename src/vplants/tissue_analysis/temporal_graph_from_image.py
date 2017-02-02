# -*- python -*-
#
#       OpenAlea.image.algo
#
#       Copyright 2012 INRIA - CIRAD - INRA
#
#       File author(s):  Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#                        Frederic Boudon <frederic.boudon@cirad.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite: http://openalea.gforge.inria.fr
################################################################################
"""This module helps to create PropertyGraph from a SpatialImage."""

import time, warnings, math, gzip
import numpy as np, copy as cp, cPickle as pickle
from os.path import exists

from openalea.image.serial.basics import SpatialImage, imread, imsave
from openalea.image.spatial_image import is2D

from openalea.container import PropertyGraph

from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis, AbstractSpatialImageAnalysis, DICT, find_wall_median_voxel, sort_boundingbox, projection_matrix


def generate_graph_topology(labels, neighborhood):
    """
    Function generating a topological/spatial graph based on neighbors detection.

    Args:
       labels: (list) - list of labels to be found in the image and added to the topological graph.
       neighborhood: (dict) - dictionary giving neighbors of each object.

    Returns:
       graph: (PropertyGraph) - the topological/spatial graph.
       label2vertex: (dict) - dictionary translating labels into vertex ids (vids).
       edges: (dict) - dictionary associating an edge id to a couple of topologically/spatially related vertex.
    """
    graph = PropertyGraph()
    vertex2label = {}
    for l in labels: vertex2label[graph.add_vertex(l)] = l
    label2vertex = dict([(j,i) for i,j in vertex2label.iteritems()])

    labelset = set(labels)
    edges = {}

    for source,targets in neighborhood.iteritems():
        if source in labelset :
            for target in targets:
                if source < target and target in labelset:
                    edges[(source,target)] = graph.add_edge(label2vertex[source],label2vertex[target])

    graph.add_vertex_property('label')
    graph.vertex_property('label').update(vertex2label)

    return graph, label2vertex, edges


def availables_spatial_properties():
    """
    Return available properties to be computed by 'temporal_graph_from_image'.
    """
    return ['boundingbox', 'volume', 'barycenter', 'L1', 'L2', 'border', 'inertia_axis', 'wall_area', 'epidermis_area', 'wall_median']


def availables_properties():
    """
    Return available properties to be computed by 'temporal_graph_from_image'.
    """
    return sorted(availables_spatial_properties())


def _graph_from_image(image, labels, background, default_properties,
                     property_as_real, ignore_cells_at_stack_margins, min_contact_area):
    """
    Construct a PropertyGraph from a SpatialImage (or equivalent) representing a segmented image.

    Args:
       image: (SpatialImage|AbstractSpatialImageAnalysis) - image containing labeled objects | analysis of an image.
       labels: (list) - list of labels to be found in the image.
        If labels is None, all labels are used.
       background: (int) - label representing background.
       default_properties: (list) - the list of name of properties to create. It should be in default_properties.
       property_as_real: (bool) - If property_as_real = True, property is in real-world units else in voxels.

    :rtype: PropertyGraph

    :Examples:

    >>> import numpy as np
    >>> image = np.array([[1, 2, 7, 7, 1, 1],
                      [1, 6, 5, 7, 3, 3],
                      [2, 2, 1, 7, 3, 3],
                      [1, 1, 1, 4, 1, 1]])

    >>> from openalea.image.algo.graph_from_image import graph_from_image
    >>> graph = graph_from_image(image)
    """

    if isinstance(image, AbstractSpatialImageAnalysis):
        analysis = image
        image = analysis.image
    else:
        try:
            analysis = SpatialImageAnalysis(image, ignoredlabels = 0, return_type = DICT, background = 1)
        except:
            analysis = SpatialImageAnalysis(image, ignoredlabels = 0, return_type = DICT)
    if ignore_cells_at_stack_margins:
        # analysis.add2ignoredlabels( analysis.cells_in_image_margins() )
        analysis.add2ignoredlabels( analysis.labels_at_stack_margins() )

    if labels is None:
        filter_label = False
        labels = list(analysis.labels())
        if background in labels : del labels[labels.index(background)]
    else:
        filter_label = True
        if isinstance(labels,int) : labels = [labels]
        # -- We don't want to have the "outer cell" (background) and "removed cells" (0) in the graph structure.
        # if 0 in labels: labels.remove(0)
        if background in labels: labels.remove(background)
        # -- If labels are provided, we ignore all others by default:
        analysis.add2ignoredlabels( set(analysis.labels()) - set(labels) )

    neighborhood = analysis.neighbors(labels, min_contact_area = min_contact_area)
    labelset = set(labels)

    graph, label2vertex, edges = generate_graph_topology(labels, neighborhood)

    # -- We want to keep the unit system of each variable
    graph.add_graph_property("units",dict())

    # if ("wall_plane_orientation" in default_properties) and ('all_wall_plane_orientation' in default_properties):
    #     default_properties.remove("wall_plane_orientation")

    if 'boundingbox' in default_properties :
        print 'Extracting boundingbox...'
        add_vertex_property_from_dictionary(graph,'boundingbox',analysis.boundingbox(labels,real=property_as_real),mlabel2vertex=label2vertex)
        #~ graph._graph_property("units").update( {"boundingbox":(u'\xb5m'if property_as_real else 'voxels')} )

    if 'volume' in default_properties and analysis.is3D():
        print 'Computing volume property...'
        add_vertex_property_from_dictionary(graph,'volume',analysis.volume(labels,real=property_as_real),mlabel2vertex=label2vertex)
        #~ graph._graph_property("units").update( {"volume":(u'\xb5m\xb3'if property_as_real else 'voxels')} )

    barycenters = None
    if 'barycenter' in default_properties :
        print 'Computing barycenter property...'
        barycenters = analysis.center_of_mass(labels,real=property_as_real)
        add_vertex_property_from_dictionary(graph,'barycenter',barycenters,mlabel2vertex=label2vertex)
        #~ graph._graph_property("units").update( {"barycenter":(u'\xb5m'if property_as_real else 'voxels')} )

    background_neighbors = set(analysis.neighbors(background))
    background_neighbors.intersection_update(labelset)
    if 'L1' in default_properties :
        print 'Generating the list of cells belonging to the first layer...'
        add_vertex_property_from_label_and_value(graph,'L1',labels,[(l in background_neighbors) for l in labels],mlabel2vertex=label2vertex)

    if 'border' in default_properties :
        print 'Generating the list of cells at the margins of the stack...'
        # border_cells = analysis.cells_in_image_margins()
        border_cells = analysis.labels_at_stack_margins()
        try: border_cells.remove(background)
        except: pass
        border_cells = set(border_cells)
        add_vertex_property_from_label_and_value(graph,'border',labels,[(l in border_cells) for l in labels],mlabel2vertex=label2vertex)

    if 'inertia_axis' in default_properties :
        print 'Computing inertia_axis property...'
        inertia_axis, inertia_values = analysis.inertia_axis(labels,barycenters)
        add_vertex_property_from_dictionary(graph,'inertia_axis',inertia_axis,mlabel2vertex=label2vertex)
        add_vertex_property_from_dictionary(graph,'inertia_values',inertia_values,mlabel2vertex=label2vertex)

    if 'wall_surface' in default_properties :
        print 'Computing wall_surface property...'
        filtered_edges, unlabelled_target, unlabelled_wall_surfaces = {}, {}, {}
        for source,targets in neighborhood.iteritems():
            if source in labelset :
                filtered_edges[source] = [ target for target in targets if source < target and target in labelset ]
                unlabelled_target[source] = [ target for target in targets if target not in labelset and target != background]
        wall_surfaces = analysis.wall_areas(filtered_edges,real=property_as_real)
        add_edge_property_from_label_property(graph,'wall_surface',wall_surfaces,mlabelpair2edge=edges)

        graph.add_vertex_property('unlabelled_wall_surface')
        for source in unlabelled_target:
            unlabelled_wall_surface = analysis.wall_areas({source:unlabelled_target[source]},real=property_as_real)
            graph.vertex_property('unlabelled_wall_surface')[label2vertex[source]] = sum(unlabelled_wall_surface.values())

        #~ graph._graph_property("units").update( {"wall_surface":('\xb5m\xb2'if property_as_real else 'voxels')} )
        #~ graph._graph_property("units").update( {"unlabelled_wall_surface":('\xb5m\xb2'if property_as_real else 'voxels')} )

    if 'epidermis_surface' in default_properties :
        print 'Computing epidermis_surface property...'
        def not_background(indices):
            a,b = indices
            if a == background:
                if b == background: raise ValueError(indices)
                else : return b
            elif b == background: return a
            else: raise ValueError(indices)
        epidermis_surfaces = analysis.cell_wall_surface(background,list(background_neighbors) ,real=property_as_real)
        epidermis_surfaces = dict([(not_background(indices),value) for indices,value in epidermis_surfaces.iteritems()])
        add_vertex_property_from_label_property(graph,'epidermis_surface',epidermis_surfaces,mlabel2vertex=label2vertex)
        #~ graph._graph_property("units").update( {"epidermis_surface":('\xb5m\xb2'if property_as_real else 'voxels')} )

    if 'wall_median' in default_properties:
        print 'Computing wall_median property...'
        dict_wall_voxels = analysis.wall_voxels_per_cells_pairs(labels, neighborhood, ignore_background=False )

        wall_median = {}
        for label_1, label_2 in dict_wall_voxels:
            #~ if dict_wall_voxels[(label_1, label_2)] == None:
                #~ if label_1 != 0:
                    #~ print "There might be something wrong between cells %d and %d" %label_1  %label_2
                #~ continue # if None we can use it.
            x,y,z = dict_wall_voxels[(label_1, label_2)]
            # compute geometric median:
            from openalea.image.algo.analysis import geometric_median, closest_from_A
            neighborhood_origin = geometric_median( np.array([list(x),list(y),list(z)]) )
            integers = np.vectorize(lambda x : int(x))
            neighborhood_origin = integers(neighborhood_origin)
            # closest points:
            pts = [tuple([int(x[i]),int(y[i]),int(z[i])]) for i in xrange(len(x))]
            min_dist = closest_from_A(neighborhood_origin, pts)
            wall_median[(label_1, label_2)] = min_dist

        edge_wall_median, unlabelled_wall_median, vertex_wall_median = {},{},{}
        for label_1, label_2 in dict_wall_voxels.keys():
            if (label_1 in graph.vertices()) and (label_2 in graph.vertices()):
                edge_wall_median[(label_1, label_2)] = wall_median[(label_1, label_2)]
            if (label_1 == 0): # no need to check `label_2` because labels are sorted in keys returned by `wall_voxels_per_cells_pairs`
                unlabelled_wall_median[label_2] = wall_median[(label_1, label_2)]
            if (label_1 == 1): # no need to check `label_2` because labels are sorted in keys returned by `wall_voxels_per_cells_pairs`
                vertex_wall_median[label_2] = wall_median[(label_1, label_2)]

        add_edge_property_from_dictionary(graph, 'wall_median', edge_wall_median)
        add_vertex_property_from_dictionary(graph, 'epidermis_wall_median', vertex_wall_median)
        add_vertex_property_from_dictionary(graph, 'unlabelled_wall_median', unlabelled_wall_median)

    return graph


spatio_temporal_properties2D = ['barycenter','boundingbox','border','L1','epidermis_area','inertia_axis']
def graph_from_image2D(image, labels, background, spatio_temporal_properties,
                     property_as_real, ignore_cells_at_stack_margins, min_contact_area):
    return _graph_from_image(image, labels, background, spatio_temporal_properties,
                            property_as_real, ignore_cells_at_stack_margins, min_contact_area)


spatio_temporal_properties3D = availables_properties()
def graph_from_image3D(image, labels, background, spatio_temporal_properties,
                     property_as_real, ignore_cells_at_stack_margins, min_contact_area):
    return _graph_from_image(image, labels, background, spatio_temporal_properties,
                            property_as_real, ignore_cells_at_stack_margins, min_contact_area)

def graph_from_image(image,
                     labels = None,
                     background = 1,
                     spatio_temporal_properties = None,
                     property_as_real = True,
                     ignore_cells_at_stack_margins = True,
                     min_contact_area = None):

    if isinstance(image, AbstractSpatialImageAnalysis):
        real_image = image.image
        if labels is None:
            labels = image.labels()
    else:
        real_image = image

    if is2D(real_image):
        if spatio_temporal_properties == None:
            spatio_temporal_properties = spatio_temporal_properties2D
        return graph_from_image2D(image, labels, background, spatio_temporal_properties,
                            property_as_real, ignore_cells_at_stack_margins, min_contact_area)
    else:
        if spatio_temporal_properties == None:
            spatio_temporal_properties = spatio_temporal_properties3D
        return graph_from_image3D(image, labels, background, spatio_temporal_properties,
                            property_as_real, ignore_cells_at_stack_margins, min_contact_area)


# def vids_handler(graph, vids, time_point = "all"):
#     """
#     Function returning an existing list of vids from the graph.
#     Use to automatically revover named groups of vertex ids (eg. 'L1', 'L2' or regions).
#        time_point` (str|list): can either be 'all' or an integer refering to an existing `time_point` within the `graph:.
#     """
#     if isinstance(vids, str):
#         if vids.lower() == 'all':
#             vids = list(graph.vertices())
#         elif (vids.lower() == 'l1') or (vids.lower() == 'l2'):
#             vids = _vids_layers(graph, vids)
#     # TODO: Add possibility to retreive vids list from 'regions'.

#     if time_point != "all":
#         vids_tp = graph.vertex_at_time(time_point)
#         vids = [v for v in vids if v in vids_tp]
#     return vids


# def _vids_layers(graph, vids):
#     """
#     Masked function used when 'vids' refer to a layer.
#     """
#     from openalea.container.interface.property_graph import PropertyError
#     try:
#         vids = [k for k,v in graph.vertex_property(vids).iteritems() if v]
#     except PropertyError as pe:
#         print pe
#         return None

#     return vids


def add_vertex_property_from_dictionary(graph, name, dictionary, mlabel2vertex = None, time_point = None, overwrite = False):
    """
        Add a vertex property with name 'name' to the graph build from an image.
        The values of the property are given as by a dictionary where keys are vertex labels.
        If overwrite is true, the property will be removed before adding the new values!
    """
    # if isinstance(graph, TemporalPropertyGraph):
    #     assert time_point is not None
    if mlabel2vertex is None:
        mlabel2vertex = label2vertex_map(graph, time_point)
    if name in graph.vertex_properties() and not overwrite:
        raise ValueError("Existing vertex property '{}'".format(name))
    if overwrite:
        print "You asked to overwrite property '{}', it will be removed first!".format(name)
        graph.remove_vertex_property(name)

    graph.add_vertex_property(name)
    graph.vertex_property(name).update( dict([(mlabel2vertex[k], dictionary[k]) for k in dictionary]) )
    return "Done."

def add_vertex_property_from_label_and_value(graph, name, labels, property_values, mlabel2vertex = None, overwrite = False):
    """
        Add a vertex property with name 'name' to the graph build from an image.
        The values of the property are given as two lists.
        First one gives the label in the image and second gives the value of the property.
        Labels are first translated in id of the graph and values are assigned to these ids in the graph
        If overwrite is true, the property will be removed before adding the new values!
    """
    if mlabel2vertex is None:
        mlabel2vertex = label2vertex_map(graph)
    if name in graph.vertex_properties() and not overwrite:
        raise ValueError("Existing vertex property '{}'".format(name))
    if overwrite:
        print "You asked to overwrite property '{}', it will be removed first!".format(name)
        graph.remove_vertex_property(name)

    graph.add_vertex_property(name)
    graph.vertex_property(name).update(dict([(mlabel2vertex[i], v) for i,v in zip(labels,property_values)]))
    return "Done."

def add_edge_property_from_dictionary(graph, name, dictionary, mlabelpair2edge = None, time_point = None, overwrite = False):
    """
        Add an edge property with name 'name' to the graph build from an image.
        The values of the property are given as by a dictionary where keys are vertex labels.
        If overwrite is true, the property will be removed before adding the new values!
    """
    if mlabelpair2edge is None:
        assert time_point is not None
        mlabelpair2edge = labelpair2edge_map(graph, time_point)
    if name in graph.edge_properties() and not overwrite:
        raise ValueError("Existing edge property '{}'".format(name))
    if overwrite:
        print "You asked to overwrite property '{}', it will be removed first!".format(name)
        graph.remove_edge_property(name)

    graph.add_edge_property(name)
    graph.edge_property(name).update( dict([(mlabelpair2edge[k], dictionary[k]) for k in dictionary]) )
    return "Done."

def add_edge_property_from_label_and_value(graph, name, label_pairs, property_values, mlabelpair2edge = None, overwrite = False):
    """
        Add an edge property with name 'name' to the graph build from an image.
        The values of the property are given as two lists.
        First one gives the pair of labels in the image that are connected and the second list gives the value of the property.
        Pairs of labels are first translated in edge ids of the graph and values are assigned to these ids in the graph
        If overwrite is true, the property will be removed before adding the new values!
    """
    if mlabelpair2edge is None:
        mlabelpair2edge = labelpair2edge_map(graph)
    if name in graph.edge_properties() and not overwrite:
        raise ValueError("Existing edge property '{}'".format(name))
    if overwrite:
        print "You asked to overwrite property '{}', it will be removed first!".format(name)
        graph.remove_edge_property(name)

    graph.add_edge_property(name)
    graph.edge_property(name).update(dict([(mlabelpair2edge[labelpair], value) for labelpair,value in zip(label_pairs,property_values)]))
    return "Done."


def retrieve_label_neighbors(SpI_Analysis, label, labelset, min_contact_area, real_area):
    """
    """
    neighbors = SpI_Analysis.neighbors(label, min_contact_area, real_area)
    return set(neighbors)&labelset

