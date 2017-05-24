# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015 INRIA - CIRAD - INRA
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

from openalea.core.plugin import PluginDef
from openalea.core.authors import gcerutti


class AppletPlugin(object):
    name_conversion = PluginDef.DROP_PLUGIN


@PluginDef
class TissueAnalysisControl(AppletPlugin):
    label = 'TissueAnalysis Controls'
    icon = ''
    authors = [gcerutti]
    implement = 'IApplet'
    __plugin__ = True

    def __call__(self):
        # Load and instantiate graphical component that actually provide feature
        from vplants.tissue_analysis.tissue_analysis_oalab.widget.tissue_analysis_panel import TissueAnalysisControlPanel
        return TissueAnalysisControlPanel


@PluginDef
class PropertyGraphControl(AppletPlugin):
    label = 'PropertyGraph Controls'
    icon = 'tissue_analysis_control.png'
    authors = [gcerutti]
    implement = 'IApplet'
    __plugin__ = True

    def __call__(self):
        # Load and instantiate graphical component that actually provide feature
        from vplants.tissue_analysis.tissue_analysis_oalab.widget.property_graph_panel import PropertyGraphControlPanel
        return PropertyGraphControlPanel



