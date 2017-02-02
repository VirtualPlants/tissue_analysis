# -*- python -*-
#
#       VPlants.tissue_analysis
#
#       Copyright 2006 - 2012 INRIA - CIRAD - INRA
#
#       File author(s): Eric MOSCARDI <eric.moscardi@sophia.inria.fr>
#                       Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
#                       Frederic BOUDON <frederic.boudon@cirad.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
################################################################################

__license__ = "Cecill-C"
__revision__ = " $Id$ "

import warnings, math, copy, gzip, time
import numpy as np, scipy.ndimage as nd, cPickle as pickle, copy as cp

from numpy.linalg import svd, norm
from os.path import exists, splitext, split

from openalea.image.serial.basics import SpatialImage, imsave

try:
    from openalea.plantgl.algo import r_neighborhood, principal_curvatures
except:
    print "PlantGL is not installed, no curvature estimation will be available !"


def dilation(slices):
    """Function dilating boundingboxes (`slices`) by one voxel."""
    return [ slice(max(0,s.start-1), s.stop+1) for s in slices ]


def dilation_by(slices, amount=2):
    """Function dilating boundingboxes (`slices`) by a given number of voxels (`amount`)."""
    return [ slice(max(0,s.start-amount), s.stop+amount) for s in slices ]


def wall(mask_img, label_id):
    """
    Function detecting wall position for a given cell-id (`label_id`) within a segmented image (`mask_img`).
    """
    img = (mask_img == label_id)
    dil = nd.binary_dilation(img)
    contact = dil - img
    return mask_img[contact]


def contact_surface(mask_img, label_id):
    """
    TODO
    """
    img = wall(mask_img,label_id)
    return set( np.unique(img) )


def real_indices(slices, resolutions):
    """
    Transform the discrete (voxels based) coordinates of the boundingbox (`slices`) into their real-world size using `resolutions`.
    
    Args:
       slices: (list) - list of slices or boundingboxes found using scipy.ndimage.find_objects;
       resolutions: (list) - length-2 (2D) or length-3 (3D) vector of float indicating the size of a voxel in real-world units;
    """
    return [ (s.start*r, s.stop*r) for s,r in zip(slices,resolutions) ]


def hollow_out_cells(image, background, remove_background = True, verbose = True):
    """
    Laplacian filter used to dectect and return an Spatial Image containing only cell walls.
    (The Laplacian of an image highlights regions of rapid intensity change.)

    Args:
       image: (SpatialImage) - Segmented image (tissu);
       background: (int) - label representing the background (to remove).

    Returns:
       m: (SpatialImage) - Spatial Image containing hollowed out cells (only walls).
    """
    if verbose: print 'Hollowing out cells... ',
    b = nd.laplace(image)
    mask = b!=0
    m = image * mask
    if remove_background:
        mask = m!=background
        m = m*mask
    if verbose: print 'Done !!'
    return m


def sort_boundingbox(boundingbox, label_1, label_2):
    """Use this to determine which label as the smaller boundingbox !"""
    assert isinstance(boundingbox, dict)
    if (not boundingbox.has_key(label_1)) and boundingbox.has_key(label_2):
        return (label_2, label_1)
    if boundingbox.has_key(label_1) and (not boundingbox.has_key(label_2)):
        return (label_1, label_2)
    if (not boundingbox.has_key(label_1)) and (not boundingbox.has_key(label_2)):
        return (None, None)

    bbox_1 = boundingbox[label_1]
    bbox_2 = boundingbox[label_2]
    vol_bbox_1 = (bbox_1[0].stop - bbox_1[0].start)*(bbox_1[1].stop - bbox_1[1].start)*(bbox_1[2].stop - bbox_1[2].start)
    vol_bbox_2 = (bbox_2[0].stop - bbox_2[0].start)*(bbox_2[1].stop - bbox_2[1].start)*(bbox_2[2].stop - bbox_2[2].start)

    return (label_1, label_2) if vol_bbox_1<vol_bbox_2 else (label_2, label_1)


def find_smallest_boundingbox(image, label_1, label_2):
    """Return the smallest boundingbox within `image` between cell-labels `label_1` & `label_2`."""
    boundingbox = nd.find_objects(image, max_label=max([label_1, label_2]))
    boundingbox = {label_1:boundingbox[label_1-1], label_2:boundingbox[label_2-1]} # we do 'label_x - 1' since 'nd.find_objects' start at '1' (and not '0') !
    label_1, label_2 = sort_boundingbox(boundingbox, label_1, label_2)
    return bbox[label_1]


def coordinates_centering3D(coordinates, mean=[]):
    """Center coordinates around their mean."""
    try:
        x,y,z = coordinates
    except:
        x,y,z = coordinates.T
    if mean == []:
        mean=np.mean(np.array([x,y,z]),1)
    # Now perform centering operation:
    x = x - mean[0]
    y = y - mean[1]
    z = z - mean[2]
    return np.array([x,y,z])

def compute_covariance_matrix(coordinates):
    """Function computing the covariance matrix of a given pointset of coordinates.

    Args:
      coordinates: (np.array) - poinset of coordinates

    Returns:
      The covariance matrix.
    """
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    if coordinates.shape[0]>3:
        coordinates = coordinates.T
    return 1./max(coordinates.shape) * np.dot(coordinates,coordinates.T)

def eigen_values_vectors(cov_matrix):
    """Function extracting the eigen vectors and associated values for a variance-covariance matrix.
    
    Args:
      cov_matrix: (np.array) - poinset of coordinates
    
    Returns:
      eig_val: (list) - lenght 3 list of sorted eigen values associated to the eigen vectors
      eig_vec: (np.array) - 3x3 np.array of eigen vectors --by rows-- associated to sorted eigen values 
    """
    assert max(cov_matrix.shape)<=3
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    decreasing_index = eig_val.argsort()[::-1]
    eig_val, eig_vec = eig_val[decreasing_index], eig_vec[:,decreasing_index] # np.linalg.eig return eigenvectors by column !!
    eig_vec = np.array(eig_vec).T # ... our standard is by rows !
    return eig_val, eig_vec

def distance(ptsA, ptsB):
    """Function computing the Euclidian distance between two points.
    Can be 2D or 3D coordinates.

    Args:
      ptsA: (list/numpy.array) - 2D/3D coordinates of point A;
      ptsB: (list/numpy.array) - 2D/3D coordinates of point B;
    
    Returns:
      The Euclidian distance between points A & B.
    """
    if len(ptsA) != len(ptsB):
        warnings.warn("It seems that the points are not in the same space!")
        return None

    if len(ptsA) == 2:
        return math.sqrt( (ptsA[0]-ptsB[0])**2+(ptsA[1]-ptsB[1])**2 )
    
    if len(ptsA) == 3:
        return math.sqrt( (ptsA[0]-ptsB[0])**2+(ptsA[1]-ptsB[1])**2+(ptsA[2]-ptsB[2])**2 )

 
def return_list_of_vectors(tensor, by_row=True):
    """Return a list of vectors from a 3x3 array read by row or columns."""
    if isinstance(tensor, dict):
        return dict([(k,return_list_of_vectors(t,by_row)) for k,t in tensor.iteritems()])
    elif isinstance(tensor, list) and tensor[0].shape == (3,3):
        return [return_list_of_vectors(t,by_row) for t in tensor]
    else:
        if by_row:
            return [tensor[v] for v in xrange(len(tensor))]
        else:
            return [tensor[:,v] for v in xrange(len(tensor))]


NPLIST, LIST, DICT = range(3)

class AbstractSpatialImageAnalysis(object):
    """
    This object is desinged to help the analysis of segmented tissues.
    It can extract 2D or 3D cellular features (volume, wall-area, ...) and the topology of a segmented tissue (SpatialImage).
    """

    def __init__(self, image, ignoredlabels = [], return_type = DICT, background = None):
        """
        ..warning :: Label features in the images are an arithmetic progression of continous integers.
        By default, we save a property or information only if it can be used by several functions.
        
        Args:
           image: (SpatialImage) - basically a two- or tri-dimensional array presenting cell label position in space (segmented image);
           ignoredlabels: (int|list) - label or list of labels to ignore when computing cell features;
           return_type: (NPLIST|LIST|DICT) - define the type used to return computed features (DICT is safer!);
           background: (int) - image-label (int) refering to the background ;
        """
        # -- We make sure that `image` is of type 'SpatialImage':
        if isinstance(image, SpatialImage):
            self.image = image
        else:
            self.image = SpatialImage(image)

        # -- We use this to avoid (when possible) computation of cell-properties (background, cells in image margins, ...)
        if isinstance(ignoredlabels, int):
            ignoredlabels = [ignoredlabels]
        self._ignoredlabels = set(ignoredlabels)

        # -- Sounds a bit paranoiac but useful !!
        if background is not None:
            if not isinstance(background,int):
                raise ValueError("The label you provided as background is not an integer !")
            if background not in self.image:
                print " WARNING!!! The background you provided has not been detected in the image !"
            self._ignoredlabels.update([background])
        else:
            warnings.warn("No value defining the background, some functionalities won't work !")

        # -- Saving useful and shared informations:
        try:
            self._voxelsize = image.voxelsize   # voxelsize in real world units (float, float, float);
        except:
            print "Could not detect voxelsize informations from the provided `image`!"
            self._voxelsize = np.ones(len(image.shape))
            print "Set by default to '{}'...".format(self._voxelsize)
        self._background = background       # image-label (int) refering to the background;
        # -- Creating masked attributes:
        self._labels = None                 # list of image-labels refering to cell-labels;
        self._bbox = None                   # list of boundingboxes (minimal cube containing a cell) sorted as self._labels;
        self._kernels = None                #
        self._neighbors = None              #
        self._cell_layer1 = None            #
        self._center_of_mass = {}           # voxel units

        # -- Saving normalized meta-informations:
        try:
            self.filepath, self.filename = split(image.info["Filename"])
        except:
            self.filepath, self.filename = None, None
        try:
            self.info = dict([(k,v) for k,v in image.info.iteritems() if k != "Filename"])
        except:
            pass

        self.return_type = return_type


    def is3D(self): return False

    def background(self): return self._background

    def ignoredlabels(self): return self._ignoredlabels

    def add2ignoredlabels(self, list2add, verbose = False):
        """
        Add labels to the ignoredlabels list (set) and update the self._labels cache.
        """
        if isinstance(list2add, int):
            list2add = [list2add]

        if verbose: print 'Adding labels', list2add,'to the list of labels to ignore...'
        self._ignoredlabels.update(list2add)
        if verbose: print 'Updating labels list...'
        self._labels = self.__labels()

    def consideronlylabels(self, list2consider, verbose = False):
        """
        Add labels to the ignoredlabels list (set) and update the self._labels cache.
        """
        if isinstance(list2consider, int):
            list2consider = [list2consider]

        toignore = set(np.unique(self.image))-set(list2consider)
        integers = np.vectorize(lambda x : int(x))
        toignore = integers(list(toignore)).tolist()


        if verbose: print 'Adding labels', toignore,'to the list of labels to ignore...'
        self._ignoredlabels.update(toignore)
        if verbose: print 'Updating labels list...'
        self._labels = self.__labels()


    def convert_return(self, values, labels = None, overide_return_type = None):
        """
        This function convert outputs of analysis functions.
        """
        tmp_save_type = copy.copy(self.return_type)
        if not overide_return_type is None:
            self.return_type = overide_return_type
        # -- In case of unique label, just return the result for this label
        if not labels is None and isinstance(labels,int):
            self.return_type = copy.copy(tmp_save_type)
            return values
        # -- return a numpy array
        elif self.return_type == NPLIST:
            self.return_type = copy.copy(tmp_save_type)
            return values
        # -- return a standard python list
        elif self.return_type == LIST:
            if isinstance(values,list):
                return values
            else:
                self.return_type = copy.copy(tmp_save_type)
                return values.tolist()
        # -- return a dictionary
        else:
            self.return_type = copy.copy(tmp_save_type)
            return dict(zip(labels,values))


    def labels(self):
        """
        Return the list of labels used.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.labels()
        [1,2,3,4,5,6,7]
        """
        if self._labels is None: self._labels = self.__labels()
        return self._labels

    def __labels(self):
        """
        Compute the actual list of labels.
        :IMPORTANT: `background` is not in the list of labels.
        """
        labels = set(np.unique(self.image))-self._ignoredlabels
        return list(map(int, labels))

    def nb_labels(self):
        """
        Return the number of labels.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.nb_labels()
        7
        """
        if self._labels is None : self._labels = self.__labels()
        return len(self._labels)

    def label_request(self, labels):
        """
        The following lines are often needed to ensure the correct format of cell-labels, as well as their presence within the image.
        """
        if isinstance(labels, int):
            if not labels in self.labels():
                print "The following id was not found within the image labels: {}".format(labels)
            labels = [labels]
        elif isinstance(labels, list):
            labels = list( set(labels) & set(self.labels()) )
            not_in_labels = list( set(labels) - set(self.labels()) )
            if not_in_labels != []:
                print "The following ids were not found within the image labels: {}".format(not_in_labels)
        elif (labels is None):
            labels = self.labels()
        elif isinstance(labels, str):
            if (labels.lower() == 'all'):
                labels = self.labels()
            elif (labels.lower() == 'l1'):
                labels = self.cell_first_layer()
            elif (labels.lower() == 'l2'):
                labels = self.cell_second_layer()
            else:
                pass
        else:
            raise ValueError("This is not usable as `labels`: {}".format(labels))

        return labels


    def center_of_mass(self, labels=None, real=True, verbose=False):
        """
        Return the center of mass of the labels.

        Args:
           labels: (int) - single label number or a sequence of label numbers of the objects to be measured.
            If labels is None, all labels are used.
           real: (bool) - If True (default), center of mass is in real-world units else in voxels.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.center_of_mass(7)
        [0.75, 2.75, 0.0]

        >>> analysis.center_of_mass([7,2])
        [[0.75, 2.75, 0.0], [1.3333333333333333, 0.66666666666666663, 0.0]]

        >>> analysis.center_of_mass()
        [[1.8, 2.2999999999999998, 0.0],
         [1.3333333333333333, 0.66666666666666663, 0.0],
         [1.5, 4.5, 0.0],
         [3.0, 3.0, 0.0],
         [1.0, 2.0, 0.0],
         [1.0, 1.0, 0.0],
         [0.75, 2.75, 0.0]]
        """
        # Check the provided `labels`:
        labels = self.label_request(labels)

        if verbose: print "Computing cells center of mass:"
        center = {}; N = len(labels); percent = 0
        for n,l in enumerate(labels):
            if verbose and (n*100/N>=percent): print "{}%...".format(percent),; percent += 5
            if verbose and (n+1==N): print "100%"
            if self._center_of_mass.has_key(l):
                center[l] = self._center_of_mass[l]
            else:
                try:
                    slices = self.boundingbox(l,real=False)
                    crop_im = self.image[slices]
                    c_o_m = np.array(nd.center_of_mass(crop_im, crop_im, index=l))
                    c_o_m = [c_o_m[i] + slice.start for i,slice in enumerate(slices)]
                except:
                    crop_im = self.image
                    c_o_m = np.array(nd.center_of_mass(crop_im, crop_im, index=l))
                self._center_of_mass[l] = c_o_m
                center[l] = c_o_m

        if real:
            center = dict([(l,np.multiply(center[l],self._voxelsize)) for l in labels])

        if len(labels)==1:
            return center[labels[0]]
        else:
            return center


    def boundingbox(self, labels = None, real = False):
        """
        Return the bounding box of a label.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.boundingbox(7)
        (slice(0, 3), slice(2, 4), slice(0, 1))

        >>> analysis.boundingbox([7,2])
        [(slice(0, 3), slice(2, 4), slice(0, 1)), (slice(0, 3), slice(0, 2), slice(0, 1))]

        >>> analysis.boundingbox()
        [(slice(0, 4), slice(0, 6), slice(0, 1)),
        (slice(0, 3), slice(0, 2), slice(0, 1)),
        (slice(1, 3), slice(4, 6), slice(0, 1)),
        (slice(3, 4), slice(3, 4), slice(0, 1)),
        (slice(1, 2), slice(2, 3), slice(0, 1)),
        (slice(1, 2), slice(1, 2), slice(0, 1)),
        (slice(0, 3), slice(2, 4), slice(0, 1))]
        """
        if labels == 0:
            return nd.find_objects(self.image==0)[0]

        if self._bbox is None:
            self._bbox = nd.find_objects(self.image)

        if labels is None:
            labels = copy.copy(self.labels())
            if self.background() is not None:
                labels.append(self.background())

        # bbox of object labelled 1 to n are stored into self._bbox. To access i-th element, we have to use i-1 index
        if isinstance (labels, list):
            bboxes = [self._bbox[i-1] for i in labels]
            if real : return self.convert_return([real_indices(bbox,self._voxelsize) for bbox in bboxes],labels)
            else : return self.convert_return(bboxes,labels)

        else :
            try:
                if real:  return real_indices(self._bbox[labels-1], self._voxelsize)
                else : return self._bbox[labels-1]
            except:
                return None


    def neighbors(self, labels=None, min_contact_area=None, real_area=True, verbose=True):
        """
        Return the list of neighbors of a label.

        :WARNING:
            If `min_contact_area` is given it should be in real world units.

        Args:
           labels: (None|int|list) - label or list of labels of which we want to return the neighbors. If none, neighbors for all labels found in self.image will be returned.
           min_contact_area: (None|int|float) - value of the min contact area threshold.
           real_area: (bool) - indicate wheter the min contact area is a real world value or a number of voxels.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.neighbors(7)
        [1, 2, 3, 4, 5]

        >>> analysis.neighbors([7,2])
        {7: [1, 2, 3, 4, 5], 2: [1, 6, 7] }

        >>> analysis.neighbors()
        {1: [2, 3, 4, 5, 6, 7],
         2: [1, 6, 7],
         3: [1, 7],
         4: [1, 7],
         5: [1, 6, 7],
         6: [1, 2, 5],
         7: [1, 2, 3, 4, 5] }
        """
        if (min_contact_area is not None) and verbose:
            if real_area:
                try: print u"Neighbors will be filtered according to a min contact area of %.2f \u03BCm\u00B2" %min_contact_area
                except: print u"Neighbors will be filtered according to a min contact area of %.2f micro m 2" %min_contact_area
            else:
                print "Neighbors will be filtered according to a min contact area of %d voxels" %min_contact_area
        if labels is None:
            return self._all_neighbors(min_contact_area, real_area)
        elif not isinstance (labels , list):
            return self._neighbors_with_mask(labels, min_contact_area, real_area)
        else:
            return self._neighbors_from_list_with_mask(labels, min_contact_area, real_area)

    def _neighbors_with_mask(self, label, min_contact_area=None, real_area=True):
        if not self._neighbors is None and label in self._neighbors.keys():
            result = self._neighbors[label]
            if  min_contact_area is None:
                return result
            else:
                return self._neighbors_filtering_by_contact_area(label, result, min_contact_area, real_area)

        try:
            slices = self.boundingbox(label)
            ex_slices = dilation(slices)
            mask_img = self.image[ex_slices]
        except:
            mask_img = self.image
        neigh = list(contact_surface(mask_img,label))
        if min_contact_area is not None:
            neigh = self._neighbors_filtering_by_contact_area(label, neigh, min_contact_area, real_area)

        return neigh

    def _neighbors_from_list_with_mask(self, labels, min_contact_area=None, real_area=True):
        if (not self._neighbors is None) and (sum([i in self._neighbors.keys() for i in labels])==len(labels)):
            result = dict([(i,self._neighbors[i]) for i in labels])
            if  min_contact_area is None:
                return result
            else:
                return self._filter_with_area(result, min_contact_area, real_area)

        edges = {}
        for label in labels:
            try:
                slices = self.boundingbox(label)
                ex_slices = dilation(slices)
                mask_img = self.image[ex_slices]
            except:
                mask_img = self.image
            neigh = list(contact_surface(mask_img,label))
            if min_contact_area is not None:
                neigh = self._neighbors_filtering_by_contact_area(label, neigh, min_contact_area, real_area)
            edges[label] = neigh

        return edges

    def _all_neighbors(self, min_contact_area=None, real_area=True):
        if not self._neighbors is None:
            result = self._neighbors
            if  min_contact_area is None:
                return result
            else:
                return self._filter_with_area(result, min_contact_area, real_area)

        edges = {} # store src, target
        slice_label = self.boundingbox()
        if self.return_type == 0 or self.return_type == 1:
            slice_label = dict( (label+1,slices) for label, slices in enumerate(slice_label))
            # label_id = label +1 because the label_id begin at 1
            # and the enumerate begin at 0.
        for label_id, slices in slice_label.items():
            # sometimes, the label doesn't exist and 'slices' is None, hence we do a try/except:
            try:
                ex_slices = dilation(slices)
                mask_img = self.image[ex_slices]
            except:
                mask_img = self.image
            neigh = list(contact_surface(mask_img,label_id))
            edges[label_id]=neigh

        self._neighbors = edges
        if min_contact_area is None:
            return edges
        else:
            return self._filter_with_area(edges, min_contact_area, real_area)

    def _filter_with_area(self, neighborhood_dictionary, min_contact_area, real_area):
        """
        Function filtering a neighborhood dictionary according to a minimal contact area between two neigbhors.

        Args:
           neighborhood_dictionary: (dict) - dictionary of neighborhood to be filtered.
           min_contact_area: (None|int|float) - value of the min contact area threshold.
           real_area: (bool) - indicate wheter the min contact area is a real world value or a number of voxels.
        """
        filtered_dict = {}
        for label in neighborhood_dictionary.keys():
            filtered_dict[label] = self._neighbors_filtering_by_contact_area(label, neighborhood_dictionary[label], min_contact_area, real_area)

        return filtered_dict

    def _neighbors_filtering_by_contact_area(self, label, neighbors, min_contact_area, real_area):
        """
        Function used to filter the returned neighbors according to a given minimal contact area between them!

        Args:
           label: (int) - label of the image to threshold by the min contact area.
           neighbors` (list) - list of neighbors of the `label: to be filtered.
           min_contact_area: (None|int|float) - value of the min contact area threshold.
           real_area: (bool) - indicate wheter the min contact area is a real world value or a number of voxels.
        """
        areas = self.cell_wall_area(label, neighbors, real_area)
        nei = cp.copy(neighbors)
        for i,j in areas.keys():
            if areas[(i,j)] < min_contact_area:
                nei.remove( i if j==label else j )

        return nei

    def neighbor_kernels(self):
        if self._kernels is None:
            if self.is3D():
                X1kernel = np.zeros((3,3,3),np.bool)
                X1kernel[:,1,1] = True
                X1kernel[0,1,1] = False
                X2kernel = np.zeros((3,3,3),np.bool)
                X2kernel[:,1,1] = True
                X2kernel[2,1,1] = False
                Y1kernel = np.zeros((3,3,3),np.bool)
                Y1kernel[1,:,1] = True
                Y1kernel[1,0,1] = False
                Y2kernel = np.zeros((3,3,3),np.bool)
                Y2kernel[1,:,1] = True
                Y2kernel[1,2,1] = False
                Z1kernel = np.zeros((3,3,3),np.bool)
                Z1kernel[1,1,:] = True
                Z1kernel[1,1,0] = False
                Z2kernel = np.zeros((3,3,3),np.bool)
                Z2kernel[1,1,:] = True
                Z2kernel[1,1,2] = False
                self._kernels = (X1kernel,X2kernel,Y1kernel,Y2kernel,Z1kernel,Z2kernel)
            else:
                X1kernel = np.zeros((3,3),np.bool)
                X1kernel[:,1] = True
                X1kernel[0,1] = False
                X2kernel = np.zeros((3,3),np.bool)
                X2kernel[:,1] = True
                X2kernel[2,1] = False
                Y1kernel = np.zeros((3,3),np.bool)
                Y1kernel[1,:] = True
                Y1kernel[1,0] = False
                Y2kernel = np.zeros((3,3),np.bool)
                Y2kernel[1,:] = True
                Y2kernel[1,2] = False
                self._kernels = (X1kernel,X2kernel,Y1kernel,Y2kernel)

        return self._kernels

    def neighbors_number(self, labels=None, min_contact_area=None, real_area=True, verbose=True):
        """
        Return the number of neigbors of each label.
        """
        nei = self.neighbors(labels, min_contact_area, real_area, verbose)
        if isinstance(nei, dict):
            return dict([(k,len(v)) for k,v in nei.iteritems()])
        else:
            return len(nei)

    def get_all_wall_binary_image(self):
        """
        Returns a binary image made of the walls positions only.
        """
        lp = nd.laplace(self.image)
        return lp/lp

    def get_voxel_face_surface(self):
        a = self._voxelsize
        if len(a)==3:
            return np.array([a[1] * a[2],a[2] * a[0],a[0] * a[1] ])
        if len(a)==2:
            return np.array([a[0],a[1]])


    def wall_voxels_between_two_cells(self, label_1, label_2, bbox=None, verbose=False):
        """
        Return the voxels coordinates defining the contact wall between two labels.

        Args:
           image: (ndarray of ints) - Array containing objects defined by labels
           label_1: (int) - object id #1
           label_2: (int) - object id #2
           bbox: (dict, optional) - If given, contain a dict of slices

        Returns:
         - xyz 3xN array.
        """

        if bbox is not None:
            if isinstance(bbox, dict):
                label_1, label_2 = sort_boundingbox(bbox, label_1, label_2)
                boundingbox = bbox[label_1]
            elif isinstance(bbox, tuple) and len(bbox)==3:
                boundingbox = bbox
            else:
                try:
                    boundingbox = find_smallest_boundingbox(self.image, label_1, label_2)
                except:
                    print "Could neither use the provided value of `bbox`, nor gess it!"
                    boundingbox = tuple([(0,s-1,None) for s in self.image.shape])
            dilated_bbox = dilation( boundingbox )
            dilated_bbox_img = self.image[dilated_bbox]
        else:
            try:
                boundingbox = find_smallest_boundingbox(self.image, label_1, label_2)
            except:
                dilated_bbox_img = self.image

        mask_img_1 = (dilated_bbox_img == label_1)
        mask_img_2 = (dilated_bbox_img == label_2)

        struct = nd.generate_binary_structure(3, 2)
        dil_1 = nd.binary_dilation(mask_img_1, structure=struct)
        dil_2 = nd.binary_dilation(mask_img_2, structure=struct)
        x,y,z = np.where( ( (dil_1 & mask_img_2) | (dil_2 & mask_img_1) ) == 1 )

        if bbox is not None:
            return np.array( (x+dilated_bbox[0].start, y+dilated_bbox[1].start, z+dilated_bbox[2].start) )
        else:
            return np.array( (x, y, z) )


    def wall_voxels_per_cell(self, label_1, bbox=None, neighbors=None, neighbors2ignore=[], verbose=False):
        """
        Return the voxels coordinates of all walls from one cell.
        There must be a contact defined between two labels, the given one and its neighbors.
        If no 'neighbors' list is provided, we first detect all neighbors of 'label_1'.

        Args:
           image: (ndarray of ints) - Array containing objects defined by labels
           label_1: (int): cell id #1.
           bbox: (dict, optional) - dictionary of slices defining bounding box for each labelled object.
           neighbors` (list, optional) - list of neighbors for the object `label_1:.
           neighbors2ignore` (list, optional) - labels of neighbors to ignore while considering separation between the object `label_1: and its neighbors. All ignored labels will be returned as 0.

        Returns:
           coord: (dict): *keys= [min(labels_1,neighbors[n]), max(labels_1,neighbors[n])]; *values= xyz 3xN array.
        """
        # -- We use the bounding box to work faster (on a smaller image)
        if isinstance(bbox,dict):
            boundingbox = bbox(label_1)
        elif (isinstance(bbox,tuple) or isinstance(bbox,list)) and isinstance(bbox[0],slice):
            boundingbox = bbox
        elif bbox is None:
            boundingbox = self.boundingbox(label_1)
        dilated_bbox = dilation(dilation( boundingbox ))
        dilated_bbox_img = self.image[dilated_bbox]

        # -- Binary mask saying where the label_1 can be found on the image.
        mask_img_1 = (dilated_bbox_img == label_1)
        struct = nd.generate_binary_structure(3, 2)
        dil_1 = nd.binary_dilation(mask_img_1, structure=struct)

        # -- We edit the neighbors list as required:
        if neighbors is None:
            neighbors = self.neighbors(label_1)
        if isinstance(neighbors,int):
            neighbors = [neighbors]
        if isinstance(neighbors,dict) and len(neighbors)!=1:
            neighborhood = neighbors
            neighbors = copy.copy(neighborhood[label_1])
        if neighbors2ignore != []:
            for nei in neighbors2ignore:
                try:
                    neighbors.remove(nei)
                except:
                    pass

        coord = {}
        neighbors_not_found = []
        for label_2 in neighbors:
            # -- Binary mask saying where the label_2 can be found on the image.
            mask_img_2 = (dilated_bbox_img == label_2)
            dil_2 = nd.binary_dilation(mask_img_2, structure=struct)
            # -- We now intersect the two dilated binary mask to find the voxels defining the contact area between two objects:
            x,y,z = np.where( ( (dil_1 & mask_img_2) | (dil_2 & mask_img_1) ) == 1 )
            if x != []:
                if label_2 not in neighbors2ignore:
                    coord[min(label_1,label_2),max(label_1,label_2)] = np.array((x+dilated_bbox[0].start, y+dilated_bbox[1].start, z+dilated_bbox[2].start))
                elif try_to_use_neighbors2ignore: # in case we want to ignore the specific position of some neighbors we replace its id by '0':
                    if are_these_labels_neighbors(neighbors2ignore, neighborhood): # we check that all neighbors to ignore are themself a set of connected neighbors!
                        if not coord.has_key((0,label_1)):
                            coord[(0,label_1)] = np.array((x+dilated_bbox[0].start, y+dilated_bbox[1].start, z+dilated_bbox[2].start))
                        else:
                            coord[(0,label_1)] = np.hstack( (coord[(0,label_1)], np.array((x+dilated_bbox[0].start, y+dilated_bbox[1].start, z+dilated_bbox[2].start))) )
                    #~ else:
                        #~ coord[(0,label_1)] = None
            else:
                if verbose:
                    print "Couldn't find a contact between neighbor cells {} and {}".format(label_1, label_2)
                neighbors_not_found.append(label_2)

        if neighbors_not_found:
            print "Some walls have not been found comparing to the `neighbors` list of {}: {}".format(label_1, neighbors_not_found)

        return coord


    def cells_walls_coords(self):
        """Return coordinates of the voxels defining a cell wall.

        This function thus returns any voxel in contact with one of different label.

        Args:
          image (SpatialImage) - Segmented image (tissu)

        Returns:
          x,y,z (list) - coordinates of the voxels defining the cell boundaries (walls).
        """
        if self.is3D():
            image = hollow_out_cells(self.image, self.background, verbose=True)
        else:
            image = copy.copy(self.image)
            image[np.where(image==self.background)] = 0

        if self.is3D():
            x,y,z = np.where(image!=0)
            return list(x), list(y), list(z)
        else:
            x,y = np.where(image!=0)
            return list(x), list(y)


    def cell_wall_area(self, label_id, neighbors, real = True):
        """
        Return the area of contact between a label and its neighbors.
        A list or a unique id can be given as neighbors.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.cell_wall_area(7,2)
        1.0
        >>> analysis.cell_wall_area(7,[2,5])
        {(2, 7): 1.0, (5, 7): 2.0}
        """

        resolution = self.get_voxel_face_surface()
        try:
            dilated_bbox =  dilation(self.boundingbox(label_id))
            dilated_bbox_img = self.image[dilated_bbox]
        except:
            #~ dilated_bbox = tuple( [slice(0,self.image.shape[i]-1) for i in xrange(len(self.image.shape))] ) #if no slice can be found we use the whole image
            dilated_bbox_img = self.image

        mask_img = (dilated_bbox_img == label_id)

        xyz_kernels = self.neighbor_kernels()

        unique_neighbor = not isinstance(neighbors,list)
        if unique_neighbor:
            neighbors = [neighbors]

        wall = {}
        for a in xrange(len(xyz_kernels)):
            dil = nd.binary_dilation(mask_img, structure=xyz_kernels[a])
            frontier = dilated_bbox_img[dil-mask_img]

            for n in neighbors:
                nb_pix = len(frontier[frontier==n])
                if real:  area = float(nb_pix*resolution[a//2])
                else : area = nb_pix
                i,j = min(label_id,n), max(label_id,n)
                wall[(i,j)] = wall.get((i,j),0.0) + area

        if unique_neighbor: return wall.itervalues().next()
        else : return wall


    def wall_areas(self, neighbors = None, real = True):
        """
        Return the area of contact between all neighbor labels.
        If neighbors is not given, it is computed first.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.wall_areas({ 1 : [2, 3], 2 : [6] })
       {(1, 2): 5.0, (1, 3): 4.0, (2, 6): 2.0 }

        >>> analysis.wall_areas()
        {(1, 2): 5.0, (1, 3): 4.0, (1, 4): 2.0, (1, 5): 1.0, (1, 6): 1.0, (1, 7): 2.0, (2, 6): 2.0, (2, 7): 1.0, (3, 7): 2, (4, 7): 1, (5, 6): 1.0, (5, 7): 2.0 }
        """
        if neighbors is None: neighbors = self.neighbors()
        areas = {}
        for label_id, lneighbors in neighbors.iteritems():
            # To avoid computing twice the same wall area, we select walls between i and j with j > i.
            neigh = [n for n in lneighbors if n > label_id]
            if len(neigh) > 0:
                lareas = self.cell_wall_area(label_id, neigh, real = real)
                for i,j in lareas.iterkeys():
                    areas[(i,j)] = areas.get((i,j),0.0) + lareas[(i,j)]
        return areas


    def cell_first_layer(self, filter_by_area = True, minimal_external_area=10, real_area=True):
        """
        Extract a list of labels corresponding to the external layer of cells.
        """
        integers = lambda l : map(int, l)
        if self._cell_layer1 is None : # _cell_layer1 contains always all the l1-cell labels.
            self._cell_layer1 = integers(self.neighbors(self.background()))

        cell_layer1 = self._cell_layer1
        if filter_by_area:
            print 
            labels_area = self.cell_wall_area(self.background(),self._cell_layer1, real_area)
            cell_layer1 = [label for label in self._cell_layer1 if ((labels_area.has_key(tuple([self.background(),label]))) and (labels_area[(self.background(),label)]>minimal_external_area))]

        return list( set(cell_layer1)-self._ignoredlabels )

    def cell_second_layer(self, filter_by_area=True, minimal_L1_area=10, real_area=True):
        """
        Extract a list of labels corresponding to the second layer of cells.
        """
        L1_neighbors=self.neighbors(self.cell_first_layer(),minimal_L1_area,real_area,True)
        l2 =set([])
        for nei in L1_neighbors.values():
            l2.update(nei)

        self._cell_layer2 = list(l2-set(self._cell_layer1)-self._ignoredlabels)
        return self._cell_layer2

    def __voxel_first_layer(self, keep_background=True):
        """
        Extract the first layer of voxels at the surface of the biological object.
        """
        print "Extracting the first layer of voxels..."
        mask_img_1 = (self.image == self.background())
        struct = nd.generate_binary_structure(3, 1)
        dil_1 = nd.binary_dilation(mask_img_1, structure=struct)

        layer = dil_1 - mask_img_1

        if keep_background:
            return self.image * layer + mask_img_1
        else:
            return self.image * layer

    def voxel_first_layer(self, keep_background=True):
        """
        Function extracting the first layer of voxels in contact with the background.
        """
        if self._voxel_layer1 is None :
            self._voxel_layer1 = self.__voxel_first_layer(keep_background)
        return self._voxel_layer1


    def wall_voxels_per_cells_pairs(self, labels=None, neighborhood=None, only_epidermis=False, ignore_background=False, min_contact_area=None, real_area=True, verbose=True):
        """
        Extract the coordinates of voxels defining the 'wall' between a pair of labels.
        :WARNING: if dimensionality = 2, only the cells belonging to the outer layer of the object will be used.

        Args:
           labels: (int|list) - label or list of labels to extract walls coordinate with its neighbors.
           neighborhood: (list|dict) - list of neighbors of label if isinstance(labels,int), if not neighborhood should be a dictionary of neighbors by labels.
           only_epidermis: (bool) - indicate if we work with the whole image or just the first layer of voxels (epidermis).
           ignore_background: (bool) - indicate whether we want to return the coordinate of the voxels defining the 'epidermis wall' (in contact with self.background()) or not.
           min_contact_area: (None|int|float) - value of the min contact area threshold.
           real_area: (bool) - indicate wheter the min contact surface is a real world value or a number of voxels.
        """
        if only_epidermis:
            image = self.voxel_first_layer(True)
        else:
            image = self.image

        compute_neighborhood=False
        if neighborhood is None:
            compute_neighborhood=True
        if isinstance(labels,list) and isinstance(neighborhood,dict):
            labels = [label for label in labels if neighborhood.has_key(label)]

        if labels is None and not only_epidermis:
            labels=self.labels()
        elif labels is None and only_epidermis:
            labels=np.unique(image)
        elif isinstance(labels,list):
            labels.sort()
            if not isinstance(neighborhood,dict):
                compute_neighborhood=True
        elif isinstance(labels,int):
            labels = [labels]
        else:
            raise ValueError("Couldn't find any labels.")

        dict_wall_voxels = {}; N = len(labels); percent = 0
        for n,label in enumerate(labels):
            if verbose and n*100/float(N) >= percent: print "{}%...".format(percent),; percent += 10
            if verbose and n+1==N: print "100%"
            # - We compute or use the neighborhood of `label`:
            if compute_neighborhood:
                neighbors = self.neighbors(label, min_contact_area, real_area)
            else:
                if isinstance(neighborhood,dict):
                    neighbors = copy.copy( neighborhood[label] )
                if isinstance(neighborhood,list):
                    neighbors = neighborhood
            # - We create a list of neighbors to ignore:
            if ignore_background:
                neighbors2ignore = [ n for n in neighbors if n not in labels ]
            else:
                neighbors2ignore = [ n for n in neighbors if n not in labels+[self.background()] ]
            # - We remove the couples of labels from which the "wall voxels" are already extracted:
            for nei in neighbors:
                if dict_wall_voxels.has_key( (min(label,nei),max(label,nei)) ):
                    neighbors.remove(nei)
            # - If there are neighbors left in the list, we extract the "wall voxels" between them and `label`:
            if neighbors != []:
                dict_wall_voxels.update(self.wall_voxels_per_cell(label, self.boundingbox(label), neighbors, neighbors2ignore, verbose=False))

        return dict_wall_voxels


    def fuse_labels_in_image(self, labels, verbose = True):
        """ Modify the image so the given labels are fused (to the min value)."""
        assert isinstance(labels, list) and len(labels) >= 2
        assert self.background() not in labels

        min_lab = min(labels)
        labels.remove(min_lab)
        N=len(labels); percent = 0
        if verbose: print "Fusing the following {} labels: {} to value '{}'.".format(N, labels, min_lab)
        for n, label in enumerate(labels):
            if verbose and n*100/float(N) >= percent: print "{}%...".format(percent),; percent += 5
            if verbose and n+1==N: print "100%"
            try:
                bbox = self.boundingbox(label)
                xyz = np.where( (self.image[bbox]) == label )
                self.image[tuple((xyz[0]+bbox[0].start, xyz[1]+bbox[1].start, xyz[2]+bbox[2].start))]=min_lab
            except:
                print "No boundingbox found for cell id #{}, skipping...".format(label)
                continue
        print "Done!"
        return None

    def remove_labels_from_image(self, labels, erase_value = 0, verbose = True):
        """
        Use remove_cell to iterate over a list of cell to remove if there is more cells to keep than to remove.
        If there is more cells to remove than to keep, we fill a "blank" image with those to keep.
        :!!!!WARNING!!!!:
        This function modify the SpatialImage 'self.image' !
        :!!!!WARNING!!!!:
        """
        #- Make sure 'labels' is a list:
        if isinstance(labels,int): labels = [labels]
        #- Make sure the background is not in the list of labels to remove!
        try: labels.remove(self.background())
        except: pass
        #- Now we can safely remove 'labels' using boundingboxes to speed-up computation and save memory:
        N=len(labels); percent = 0
        if verbose: print "Removing", N, "cell-labels."
        for n, label in enumerate(labels):
            if verbose and n*100/float(N) >= percent: print "{}%...".format(percent),; percent += 5
            if verbose and n+1==N: print "100%"
            try:
                xyz = np.where( (self.image[self.boundingbox(label)]) == label )
                self.image[tuple((xyz[0]+self.boundingbox(label)[0].start, xyz[1]+self.boundingbox(label)[1].start, xyz[2]+self.boundingbox(label)[2].start))]=erase_value
            except:
                print "No boundingbox found for cell id #{}, skipping...".format(label)
                continue
        #- We now update the 'self._ignoredlabels' labels list:
        self._ignoredlabels.update([erase_value])
        [self._ignoredlabels.discard(label) for label in labels]

        if verbose: print 'Done !!'


    def remove_stack_margin_labels_from_image(self, erase_value = 0, voxel_distance_from_margin=5, verbose = True):
        """
        :!!!!WARNING!!!!:
        This function modify the SpatialImage 'self.image' !
        :!!!!WARNING!!!!:
        Function removing cells at the margins, because most probably partially acquired.
        """
        if verbose: print "Deleting cells at the margins of the stack from 'self.image'..."
        self.remove_labels_from_image(self.labels_at_stack_margins(voxel_distance_from_margin), erase_value, verbose)


class SpatialImageAnalysis3D(AbstractSpatialImageAnalysis):
    """
    Class dedicated to 3D objects.
    """

    def __init__(self, image, ignoredlabels = [], return_type = DICT, background = None):
        AbstractSpatialImageAnalysis.__init__(self, image, ignoredlabels, return_type, background)
        self._voxel_layer1 = None
        self.principal_curvatures = {}
        self.principal_curvatures_normal = {}
        self.principal_curvatures_directions = {}
        self.principal_curvatures_origin = {}
        self.curvatures_tensor = {}
        self.external_wall_geometric_median = {}
        self.epidermis_wall_median_voxel = {}

    def is3D(self): return True

    def volume(self, labels = None, real = True):
        """
        Return the volume of the labels.

        Args:
           labels: (int) - single label number or a sequence of
            label numbers of the objects to be measured.
            If labels is None, all labels are used.

           real: (bool) - If real = True, volume is in real-world units else in voxels.

        :Examples:

        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
        >>> analysis = SpatialImageAnalysis(a)

        >>> analysis.volume(7)
        4.0

        >>> analysis.volume([7,2])
        [4.0, 3.0]

        >>> analysis.volume()
        [10.0, 3.0, 4.0, 1.0, 1.0, 1.0, 4.0]
        """
        # Check the provided `labels`:
        labels = self.label_request(labels)

        volume = nd.sum(np.ones_like(self.image), self.image, index=np.int16(labels))
        # convert to real-world units if asked:
        if real is True:
            if self.image.ndim == 2:
                volume = np.multiply(volume,(self._voxelsize[0]*self._voxelsize[1]))
            elif self.image.ndim == 3:
                volume = np.multiply(volume,(self._voxelsize[0]*self._voxelsize[1]*self._voxelsize[2]))
            volume.tolist()

        if not isinstance(labels, int):
            return self.convert_return(volume, labels)
        else:
            return volume


    def inertia_axis(self, labels = None, real = True, verbose=False):
        """
        Return the inertia axis of cells, also called the shape main axis.
        Return 3 (3D-oriented) vectors by rows and 3 (length) values.
        """
        # Check the provided `labels`:
        labels = self.label_request(labels)

        # results
        inertia_eig_vec = []
        inertia_eig_val = []
        N = len(labels); percent=0
        for i,label in enumerate(labels):
            if verbose and i*100/float(N) >= percent: print "{}%...".format(percent),; percent += 10
            if verbose and i+1==N: print "100%"
            slices = self.boundingbox(label, real=False)
            center = copy.copy(self.center_of_mass(label, real=False))
            # project center into the slices sub_image coordinate
            if slices is not None:
                for i,slice in enumerate(slices):
                    center[i] = center[i] - slice.start
                label_image = (self.image[slices] == label)
            else:
                print 'No boundingbox found for label {}'.format(label)
                label_image = (self.image == label)

            # compute the indices of voxel with adequate label
            xyz = label_image.nonzero()
            if len(xyz)==0:
                continue # obviously no reasons to go further !
            coord = coordinates_centering3D(xyz, center)
            # compute the variance-covariance matrix (1/N*P.P^T):
            cov = compute_covariance_matrix(coord)
            # Find the eigen values and vectors.
            eig_val, eig_vec = eigen_values_vectors(cov)
            # convert to real-world units if asked:
            if real:
                for i in xrange(3):
                    eig_val[i] *= np.linalg.norm( np.multiply(eig_vec[i],self._voxelsize) )

            inertia_eig_vec.append(eig_vec)
            inertia_eig_val.append(eig_val)

        if len(labels)==1 :
            return return_list_of_vectors(inertia_eig_vec[0]), inertia_eig_val[0]
        else:
            return self.convert_return(return_list_of_vectors(inertia_eig_vec),labels), self.convert_return(inertia_eig_val,labels)


    def reduced_inertia_axis(self, labels = None, real = True, verbose=False):
        """
        Return the REDUCED (centered coordinates standardized) inertia axis of cells, also called the shape main axis.
        Return 3 (3D-oriented) vectors by rows and 3 (length) values.
        """
        # Check the provided `labels`:
        labels = self.label_request(labels)

        # results
        inertia_eig_vec = []
        inertia_eig_val = []
        N = len(labels); percent=0
        for i,label in enumerate(labels):
            if verbose and i*100/float(N) >= percent: print "{}%...".format(percent),; percent += 10
            if verbose and i+1==N: print "100%"
            slices = self.boundingbox(label, real=False)
            center = copy.copy(self.center_of_mass(label, real=False))
            # project center into the slices sub_image coordinate
            if slices is not None:
                for i,slice in enumerate(slices):
                    center[i] = center[i] - slice.start
                label_image = (self.image[slices] == label)
            else:
                print 'No boundingbox found for label {}'.format(label)
                label_image = (self.image == label)

            # compute the indices of voxel with adequate label
            xyz = label_image.nonzero()
            if len(xyz)==0:
                continue # obviously no reasons to go further !
            coord = coordinates_centering3D(xyz, center)
            # compute the variance-covariance matrix (1/N*P.P^T):
            cov = compute_covariance_matrix(coord)
            # Find the eigen values and vectors.
            eig_val, eig_vec = eigen_values_vectors(cov)
            # convert to real-world units if asked:
            if real:
                for i in xrange(3):
                    eig_val[i] *= np.linalg.norm( np.multiply(eig_vec[i],self._voxelsize) )

            inertia_eig_vec.append(eig_vec)
            inertia_eig_val.append(eig_val)

        if len(labels)==1 :
            return return_list_of_vectors(inertia_eig_vec[0],by_row=1), inertia_eig_val[0]
        else:
            return self.convert_return(return_list_of_vectors(inertia_eig_vec,by_row=1),labels), self.convert_return(inertia_eig_val,labels)


    def labels_at_stack_margins(self, voxel_distance_from_margin=5):
        """
        Return a list of cells in contact with the margins of the stack (SpatialImage).
        All ids within a defined (5 by default) voxel distance form the margins will be used to define cells as 'in image margins'.
        """
        vx_dist=voxel_distance_from_margin
        margins = []
        margins.extend(np.unique(self.image[:vx_dist,:,:]))
        margins.extend(np.unique(self.image[-vx_dist:,:,:]))
        margins.extend(np.unique(self.image[:,:vx_dist,:]))
        margins.extend(np.unique(self.image[:,-vx_dist:,:]))
        margins.extend(np.unique(self.image[:,:,:vx_dist]))
        margins.extend(np.unique(self.image[:,:,-vx_dist:]))

        return list(set(margins)-set([self._background]))


    def region_boundingbox(self, labels):
        """
        This function return a boundingbox of a region including all cells (provided by `labels`).

        Args:
           labels: (list): list of cells ids;
        Returns:
        # - [x_start,y_start,z_start,x_stop,y_stop,z_stop] # 09.12.15: changed to the next line to match boundingbox order (as returned by scipy.ndimage.find_object !!
         - (slice(x_start,x_stop), slice(y_start,y_stop), slice(z_start,z_stop))
        """
        if isinstance(labels,list) and len(labels) == 1:
            return self.boundingbox(labels[0])
        if isinstance(labels,int):
            return self.boundingbox(labels)

        dict_slices = self.boundingbox(labels)
        #-- We start by making sure that all cells have an entry (key) in `dict_slices`:
        not_found=[]
        for c in labels:
            if c not in dict_slices.keys():
                not_found.append(c)
        if len(not_found)!=0:
            warnings.warn('You have asked for unknown cells labels: '+" ".join([str(k) for k in not_found]))

        #-- We now define a slice for the region including all cells:
        x_start,y_start,z_start,x_stop,y_stop,z_stop=np.inf,np.inf,np.inf,0,0,0
        for c in labels:
            x,y,z=dict_slices[c]
            x_start=min(x.start,x_start)
            y_start=min(y.start,y_start)
            z_start=min(z.start,z_start)
            x_stop=max(x.stop,x_stop)
            y_stop=max(y.stop,y_stop)
            z_stop=max(z.stop,z_stop)

        return (slice(x_start,x_stop), slice(y_start,y_stop), slice(z_start,z_stop))


    def cells_voxel_layer(self, labels, region_boundingbox = False, single_frame = False):
        """
        This function extract the first layer of voxel surrounding a cell defined by `label`
        Args:
           label: (int|list) - cell-label for which we want to extract the first layer of voxel;
           region_boundingbox: (bool) - if True, consider a boundingbox surrounding all labels, instead of each label alone.
           single_frame: (bool) - if True, return only one array with all voxels position defining cell walls.
        :Output:
         returns a binary image: 1 where the cell-label of interest is, 0 elsewhere
        """
        if isinstance(labels,int):
            labels = [labels]
        if single_frame:
            region_boundingbox=True

        if not isinstance(region_boundingbox,bool):
            if sum([isinstance(s,slice) for s in region_boundingbox])==3:
                bbox = region_boundingbox
            else:
                print "TypeError: Wong type for 'region_boundingbox', should either be bool or la tuple of slices"
                return None
        elif isinstance(region_boundingbox,bool) and region_boundingbox:
            bbox = self.region_boundingbox(labels)
        else:
            bboxes = self.boundingbox(labels, real=False)
        
        # Generate the smaller eroding structure possible:
        struct = nd.generate_binary_structure(3, 2)
        if single_frame:
            vox_layer = np.zeros_like(self.image[bbox], dtype=int)
        else:
            vox_layer = {}
        for clabel in labels:
            if region_boundingbox:
                bbox_im = self.image[bbox]
            else:
                bbox_im = self.image[bboxes[clabel]]
            # Creating a mask (1 where the cell-label of interest is, 0 elsewhere):
            mask_bbox_im = (bbox_im == clabel)
            # Erode the cell using the structure:
            eroded_mask_bbox_im = nd.binary_erosion(mask_bbox_im, structure=struct)
            if single_frame:
                vox_layer += np.array(mask_bbox_im - eroded_mask_bbox_im, dtype=int)
            else:
                vox_layer[clabel] = np.array(mask_bbox_im - eroded_mask_bbox_im, dtype=int)
            
        if len(labels)==1:
            return vox_layer[clabel]
        else:
            return vox_layer


def outliers_exclusion( data, std_multiplier = 3, display_data_plot = False):
    """
    Return a list or a dict (same type as `data`) cleaned out of outliers.
    Outliers are detected according to a distance from standard deviation.
    """
    from numpy import std,mean
    tmp = copy.deepcopy(data)
    if isinstance(data,list):
        borne = mean(tmp) + std_multiplier*std(tmp)
        N = len(tmp)
        n=0
        while n < N:
            if (tmp[n]>borne) or (tmp[n]<-borne):
                tmp.pop(n)
                N = len(tmp)
            else:
                n+=1
    if isinstance(data,dict):
        borne = mean(tmp.values()) + std_multiplier*std(tmp.values())
        for n in data:
            if (tmp[n]>borne) or (tmp[n]<-borne):
                tmp.pop(n)
    if display_data_plot:
        import matplotlib.pyplot as plt
        if isinstance(data,list):
            plt.plot( data )
            plt.plot( tmp )
        plt.show()
        if isinstance(data,dict):
            plt.plot( data.values() )
            plt.plot( tmp.values() )
        plt.show()
    return tmp


def vector_correlation(vect1,vect2):
    """
    Compute correlation between two vector, which is the the cosine of the angle between two vectors in Euclidean space of any number of dimensions.
    The dot product is directly related to the cosine of the angle between two vectors if they are normed !!!
    """
    # -- We make sure that we have normed vectors.
    from numpy.linalg import norm
    if (np.round(norm(vect1)) != 1.):
        vect1 = vect1/norm(vect1)
    if (np.round(norm(vect2)) != 1.):
        vect2 = vect2/norm(vect2)

    return np.round(np.dot(vect1,vect2),3)


def find_wall_median_voxel(dict_wall_voxels, labels2exclude=[], return_id=True, verbose=True):
    """Finds the voxel closest to the geometrical median of a voxels point set.
    
    This typically search for the median voxel of wall.
    It does so using a dictionary of wall-defining voxels coordinates.
    
    Args:
      dict_wall_voxels: (dict) a dictionary with labelpairs (neighbors) as keys and the xyz wall coordinates as values.
      labels2exclude: (list) allow to define a set of label for wich we will ignore the median voxel search (ex: a list of area-filtered neighbors).
      return_id: (bool) define if the fuction return the median voxel id (index in the poinset) or coodinate.
      
    Returns:
      By default returns the index of the geometrical median point of the xyz pointset (return_id=True).
      If `return_id` is set to False, it returns the actual coordiantes of the median voxel.
    """
    from numpy import ndarray

    if isinstance(labels2exclude,int):
        labels2exclude = [labels2exclude]

    if isinstance(dict_wall_voxels, dict):
        wall_median = {}; N = len(dict_wall_voxels); percent = 0
        for n,(label_1, label_2) in enumerate(dict_wall_voxels):
            if verbose and n*100/float(N) >= percent: print "{}%...".format(percent),; percent += 10
            if verbose and n+1==N: print "100%"
            if label_1 in labels2exclude or label_2 in labels2exclude:
                continue
            xyz = np.array(dict_wall_voxels[(label_1, label_2)])
            if xyz.shape[0] == 3:
                xyz = xyz.T
            median_vox_id = _find_wall_median_voxel(xyz)
            if return_id:
                wall_median[(label_1, label_2)] = median_vox_id
            else:
                wall_median[(label_1, label_2)] = xyz[median_vox_id]

        if len(dict_wall_voxels) == 1:
            return wall_median.values()[0]
        else:
            return wall_median

    if isinstance(dict_wall_voxels, ndarray):
        xyz = dict_wall_voxels
        if xyz.shape[0] == 3:
            xyz = np.array(xyz).T
        median_vox_id = _find_wall_median_voxel(xyz)

        if return_id:
            return median_vox_id
        else:
            return xyz[median_vox_id]
    else:
        return "Failed to recognise the type of data."

def _find_wall_median_voxel(array):
    """sub function searching for median voxel using PlantGL functions.
    
    If there is less than a hundred coordinate we use an exact search thanks to `pointset_median`.
    Else we do an approximate search using `approx_pointset_median`.
    
    Args:
      array: a 3xN array of coordinates.
    
    Returns:
      the id of the median voxel among the voxel pointset.
    
    Example:
    >>> ar = np.array([[0,0,0], [0,1,0], [0,2,0], [0,3,0], [0,4,0]])
    >>> _find_wall_median_voxel(ar)
    >>> 2
    """
    from openalea.plantgl.math import Vector3
    from openalea.plantgl.algo import approx_pointset_median, pointset_median
    # Need an array with 3D coordinates as rows:
    if array.shape[0] == 3:
        array = array.T
    # Coordinates `Vector3` conversion:
    xyz = [Vector3(list([float(i) for i in k])) for k in array]
    # Compute geometric median:
    if len(xyz) <= 100:
        median_vox_id = pointset_median(xyz)
    else:
        median_vox_id = approx_pointset_median(xyz)

    return median_vox_id


def geometric_median(X, numIter = 200):
    """
    Compute the geometric median of a point sample.
    The geometric median coordinates will be expressed in the Spatial Image reference system (not in real world metrics).
    We use the Weiszfeld's algorithm (http://en.wikipedia.org/wiki/Geometric_median)

    Args:
       X: (list|np.array) - voxels coordinate (3xN matrix)
       numIter: (int) - limit the length of the search for global optimum

    Returns:
     - np.array((x,y,z)): geometric median of the coordinates;
    """
    # -- Initialising 'median' to the centroid
    y = np.mean(X,1)
    # -- If the init point is in the set of points, we shift it:
    while (y[0] in X[0]) and (y[1] in X[1]) and (y[2] in X[2]):
        y+=0.1

    convergence=False # boolean testing the convergence toward a global optimum
    dist=[] # list recording the distance evolution

    # -- Minimizing the sum of the squares of the distances between each points in 'X' and the median.
    i=0
    while ( (not convergence) and (i < numIter) ):
        num_x, num_y, num_z = 0.0, 0.0, 0.0
        denum = 0.0
        m = X.shape[1]
        d = 0
        for j in range(0,m):
            div = math.sqrt( (X[0,j]-y[0])**2 + (X[1,j]-y[1])**2 + (X[2,j]-y[2])**2 )
            num_x += X[0,j] / div
            num_y += X[1,j] / div
            num_z += X[2,j] / div
            denum += 1./div
            d += div**2 # distance (to the median) to miminize
        dist.append(d) # update of the distance evolution

        if denum == 0.:
            warnings.warn( "Couldn't compute a geometric median, please check your data!" )
            return [0,0,0]

        y = [num_x/denum, num_y/denum, num_z/denum] # update to the new value of the median
        if i > 3:
            convergence=(abs(dist[i]-dist[i-2])<0.1) # we test the convergence over three steps for stability
            #~ print abs(dist[i]-dist[i-2]), convergence
        i += 1
    if i == numIter:
        raise ValueError( "The Weiszfeld's algoritm did not converged after"+str(numIter)+"iterations !!!!!!!!!" )
    # -- When convergence or iterations limit is reached we assume that we found the median.

    return np.array(y)


def are_these_labels_neighbors(labels, neighborhood):
    """
    This function allows you to make sure the provided labels are all connected neighbors according to a known neighborhood.
    """
    intersection=set()
    for label in labels:
        try:
            inter = set(neighborhood[label])&set(labels) # it's possible that `neighborhood` does not have key `label`
        except:
            inter = set()
        if inter == set(labels)-set([label]):
            return True
        if inter != set():
            intersection.update(inter)

    if intersection == set(labels):
        return True
    else:
        return False


def SpatialImageAnalysis(image, *args, **kwd):
    """
    Constructeur. Detect automatically if the image is 2D or 3D.
    For 3DS (surfacic 3D) a keyword-argument 'surf3D=True' should be passed.
    """
    # -- If 'image' is a string, it should relate to the filename and we try to load it using imread:
    if isinstance(image, str):
        from openalea.image.serial.basics import imread
        image = imread(image)
    #~ print args, kwd
    assert len(image.shape) in [2,3]

    # -- Check if the image is 2D
    if len(image.shape) == 2 or image.shape[2] == 1:
        return SpatialImageAnalysis2D(image, *args, **kwd)
    # -- Else it's considered as a 3D image.
    else:
        return SpatialImageAnalysis3D(image, *args, **kwd)


def read_id_list( filename, sep='\n' ):
    """
    Read a *.txt file containing a list of ids separated by `sep`.
    """
    f = open(filename, 'r')
    r = f.read()

    k = r.split(sep)

    list_cell = []
    for c in k:
        if c != '':
            list_cell.append(int(c))

    return list_cell


def save_id_list(id_list, filename, sep='\n' ):
    """
    Read a *.txt file containing a list of ids separated by `sep`.
    """
    f = open(filename, 'w')
    for k in id_list:
        f.write(str(k))
        f.write(sep)

    f.close()


def projection_matrix(point_set, subspace_rank = 2):
    """
    Compute the projection matrix of a set of point depending on the subspace rank.

    Args:
     - point_set (np.array): list of coordinates of shape (n_point, init_dim).
     - dimension_reduction (int) : the dimension reduction to apply
    """
    point_set = np.array(point_set)
    nb_coord = point_set.shape[0]
    init_dim = point_set.shape[1]
    assert init_dim > subspace_rank
    assert subspace_rank > 0

    centroid = point_set.mean(axis=0)
    if sum(centroid) != 0:
        # - Compute the centered matrix:
        centered_point_set = point_set - centroid
    else:
        centered_point_set = point_set

    # -- Compute the Singular Value Decomposition (SVD) of centered coordinates:
    U,D,V = svd(centered_point_set, full_matrices=False)
    V = V.T

    # -- Compute the projection matrix:
    H = np.dot(V[:,0:subspace_rank], V[:,0:subspace_rank].T)

    return H

def random_color_dict(list_cell, alea_range=None):
    """
    Generate a dict where keys -from a given list `list_cell`- receive a random integer from the list as value.
    """
    import random
    if isinstance(alea_range,int):
        return dict(zip( list_cell, [random.randint(0, alea_range) for k in xrange(len(list_cell))] ))
    elif isinstance(alea_range,list) and (len(alea_range)==2):
        return dict(zip( list_cell, [random.randint(alea_range[0], alea_range[1]) for k in xrange(len(list_cell))] ))
    else:
        return dict(zip( list_cell, [random.randint(0, 255) for k in xrange(len(list_cell))] ))


def geometric_L1(spia):
    """
    """
    background = spia._background
    L1_labels = spia.cell_first_layer()
    L1_cells_bary = spia.center_of_mass(L1_labels, verbose=True)

    background_neighbors = spia.neighbors(L1_labels, min_contact_area=10., real_area=True)
    background_neighbors = set(background_neighbors) & set(L1_labels)
    L1_cells_bboxes = spia.boundingbox(L1_labels)

    print "-- Searching for the median voxel of each epidermis wall ..."
    dict_wall_voxels, epidermis_wall_median, median2bary_dist = {}, {}, {}
    for label_2 in background_neighbors:
        dict_wall_voxels[(background,label_2)] = wall_voxels_between_two_cells(spia.image, background, label_2, bbox = L1_cells_bboxes[label_2], verbose = False)
        epidermis_wall_median[label_2] = find_wall_median_voxel(dict_wall_voxels[(background,label_2)], verbose = False)
        median2bary_dist[label_2] = distance(L1_cells_bary[label_2], epidermis_wall_median[label_2])

    return median2bary_dist, epidermis_wall_median, L1_cells_bary

