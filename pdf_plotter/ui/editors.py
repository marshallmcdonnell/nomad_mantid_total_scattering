
import os

from traits.api \
    import Str, Instance, DelegatesTo

from traitsui.api \
    import TreeEditor, TreeNode, TreeNodeObject, ObjectTreeNode,  \
    View, Group, Menu, Action

from pdf_plotter.ui.models \
    import Dataset, CorrectedDatasets, Measurement, Experiment


class RootNode(TreeNode):

    # List of object classes the node applies to
    node_for = [Experiment]

    # Automatically open the children underneath the node
    auto_open = True

    # Specify children of node
    children = ''

    # Label of the node (this is an attribute of the class in 'node_for')
    label = '=Experiments'

    # View for the node
    view = View(Group('title', orientation='vertical', show_left=False))


class ExperimentNode(TreeNode):

    # List of object classes the node applies to
    node_for = [Experiment]

    # Automatically open the children underneath the node
    auto_open = True

    # Specify children of node (this is an attribute of the class in
    # 'node_for')
    children = 'measurements'

    # Label of the node
    label = 'title'

    # View for the node
    view = View()

    # Class of node to add
    add = [Measurement]


class MeasurementNode(TreeNode):

    # List of object classes the node applies to
    node_for = [Measurement]

    # Automatically open the children underneath the node
    auto_open = False

    # Label of the node (this is an attribute of the class in 'node_for')
    label = 'title'

    # View for the node
    view = View(Group('title', orientation='vertical', show_left=True))

    # Class of node to add
    add = [CorrectedDatasets]


    # Must specify we have children since 'children' not defined makes this only a leaf
    def allows_children( self, object ):
        return True

    # Override to make the correct TreeNodes from the children Models of the Object (Measurement)
    def get_children( self, object ):
        nodes = [ CorrectedDatasetsNode(corrected_datasets=cd) for cd in object.corrected_datasets ]
        return nodes

class CorrectedDatasetsNode(TreeNodeObject):

    # The CorrectedDatasets model this tree node represents
    corrected_datasets = Instance(CorrectedDatasets())

    title = DelegatesTo('corrected_datasets')

    # Image icon use for node icon (i.e. 'img.png')
    icon = Str

    # Path for finding the icon images specified in icon
    icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','images')


    # Map of corrected_datasets 'title' trait -> icon image
    sample_title2image = {
        'sample_and_container' : 'sample_and_container.png',
        'sample_normalized' : 'sample_normalized.png',
        'sample_minus_back' : 'sample_minus_back.png',
        'sample_minus_back_normalized' : 'sample_minus_back_normalized.png',
        'sample_minus_back_normalized_ms_abs_corrected' : 'sample_minus_back_normalized_ms_abs_corrected.png',
        'sample_minus_back_normalized_ms_abs_corrected_norm_by_atoms' : 'sample_minus_back_normalized_ms_abs_corrected_norm_by_atoms.png',
        'sample_minus_back_normalized_ms_abs_corrected_norm_by_atoms_multiply_by_vanSelfScat' : 'sample_minus_back_normalized_ms_abs_corrected_norm_by_atoms_multiply_by_vanSelfScat.png',
        'sample_minus_back_normalized_ms_abs_corrected_norm_by_atoms_multiply_by_vanSelfScat_placzek_corrected' : 'sample_minus_back_normalized_ms_abs_corrected_norm_by_atoms_multiply_by_vanSelfScat_placzek_corrected.png',
    }

    vanadium_title2image = {
        'vanadium_and_background' : 'vanadium_and_background.png',
        'vanadium_minus_back' : 'vanadium_minus_back.png',
        'vanadium_minus_back_ms_abs_corrected' : 'vanadium_minus_back_ms_abs_corrected.png',
        'vanadium_minus_back_ms_abs_corrected_with_peaks' : 'vanadium_minus_back_ms_abs_corrected_with_peaks.png',
        'vanadium_minus_back_ms_abs_corrected_peaks_stripped' : 'vanadium_minus_back_ms_abs_corrected_peaks_stripped.png',
        'vanadium_minus_back_ms_abs_corrected_peaks_stripped_smoothed' : 'vanadium_minus_back_ms_abs_corrected_peaks_stripped_smoothed.png',
        'vanadium_minus_back_ms_abs_corrected_peaks_stripped_smoothed_placzek_corrected' : 'vanadium_minus_back_ms_abs_corrected_peaks_stripped_smoothed_placzek_corrected.png',
    }

    container_title2image = {
        'container' : 'container.png',
        'container_normalized' : 'container_normalized.png',
        'container_minus_back' : 'container_minus_back.png',
        'container_minus_back_normalized' : 'container_minus_back_normalized.png',
    }

    container_bckg_title2image = {
        'container_background' : 'container_background.png',
        'container_background_normalized' : 'container_background_normalized.png',
    }

    vanadium_bckg_title2image = {
        'vanadium_background' : 'vanadium_background.png',
        'vanadium_background_normalized' : 'vanadium_background_normalized.png',
    }

    correction_title2image = {
        'sample_placzek' : 'sample_placzek.png',
        'vanadium_placzek' : 'vanadium_placzek.png'
    }

    title2image = dict()
    title2image.update(sample_title2image)
    title2image.update(container_title2image)
    title2image.update(vanadium_title2image)
    title2image.update(container_bckg_title2image)
    title2image.update(vanadium_bckg_title2image)
    title2image.update(correction_title2image)

    def tno_get_label(self, node):
        #return "" # hide label
        return self.corrected_datasets.title

    def tno_has_children(self, node):
        return True

    def tno_get_children(self, node):
        return list(self.corrected_datasets.datasets)

    def tno_get_icon(self, node, state):
        if self.corrected_datasets.title in self.title2image:
            return self.title2image[ self.corrected_datasets.title ]
        else:
            return '<group>'

    def tno_get_icon_path(self, node):
        return self.icon_path

    def tno_select(self, node):
        """ Handles an object being selected.
        """
        print(node, self)
        if node.on_select is not None:
            node.on_select(self)
            return None

        return True


class CorrectedDatasetsObjectTreeNode(ObjectTreeNode):

    # List of object classes the node applies to
    node_for = [CorrectedDatasetsNode]

    # Specify children of node (this is an attribute of the class in
    # 'node_for')
    children = 'datasets'

    # Automatically open the children underneath the node
    auto_open = False

    # Label of the node (this is an attribute of the class in 'node_for')
    label = 'title'

    # View for the node
    view = View(Group('title', orientation='vertical', show_left=False))

    # Class of node to add
    add = [Dataset]

class DatasetNode(TreeNode):

    # List of object classes the node applies to
    node_for = [Dataset]

    # Automatically open the children underneath the node
    auto_open = False

    # Label of the node (this is an attribute of the class in 'node_for')
    label = 'title'

    # Menu
    menu = Menu(Action(name="Test...",
                       action="handler.get_measurement(editor,object)"))

    # View for the node
    view = View()


ExperimentTreeEditor = TreeEditor(
    nodes=[
        RootNode(),
        ExperimentNode(),
        MeasurementNode(),
        CorrectedDatasetsObjectTreeNode(),
        DatasetNode(),
    ],
    icon_size=(40,40),
    selected='selected',
    editable=False,
)
