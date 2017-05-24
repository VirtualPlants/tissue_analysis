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
from openalea.image.spatial_image import SpatialImage

from openalea.container import array_dict

from copy import copy, deepcopy

def spatial_image_analysis_to_spatial_image(input_sia, property_name=None, labels=None):
    """
    """

    sia = deepcopy(input_sia)

    img_labels = sia.labels()
    if labels is not None:
        labels_to_remove = set(img_labels) - set(list(labels))
        sia.remove_labels_from_image(labels_to_remove)

    img_labels = sia.labels()
    segmented_img = deepcopy(sia.image)
    background = sia.background()

    print background

    if property_name == 'volume':
        img_volumes = sia.volume(img_labels)
        if isinstance(img_volumes,np.ndarray) or isinstance(img_volumes,list):
            img_volumes = array_dict(img_volumes, keys=img_labels)
        elif isinstance(img_volumes,dict):
            img_volumes = array_dict(img_volumes)

        property_img = SpatialImage(img_volumes.values(segmented_img).astype(np.uint16),resolution=segmented_img.resolution)
        property_img[segmented_img == background] = background

    else:
        property_img = segmented_img

    return property_img