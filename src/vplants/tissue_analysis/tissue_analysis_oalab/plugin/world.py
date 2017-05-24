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

__revision__ = ""

from openalea.core.plugin import PluginDef
from openalea.core.authors import gcerutti


class WorldPlugin(object):
    implement = 'IWorldHandler'


@PluginDef
class PropertyImageHandlerPlugin(WorldPlugin):
    label = 'PropertyImage Handler'
    icon = 'topomesh_control.png'
    authors = [gcerutti]
    __plugin__ = True

    def __call__(self):
        from vplants.tissue_analysis.tissue_analysis_oalab.world.property_spatial_image_handler import PropertyImageHandler
        return PropertyImageHandler
