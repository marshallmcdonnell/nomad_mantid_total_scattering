from models \
    import Experiment, Measurement, CorrectedDatasets, Dataset

from traitsui.api \
    import TreeEditor, TreeNode, View, Group


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
    auto_open = True

    # Specify children of node (this is an attribute of the class in
    # 'node_for')
    children = 'corrected_datasets'

    # Label of the node (this is an attribute of the class in 'node_for')
    label = 'title'

    # View for the node
    view = View(Group('title', orientation='vertical', show_left=True))

    # Class of node to add
    add = [CorrectedDatasets]

class CorrectedDatasetsNode(TreeNode):

    # List of object classes the node applies to
    node_for = [CorrectedDatasets]

    # Automatically open the children underneath the node
    auto_open = False 

    # Specify children of node (this is an attribute of the class in
    # 'node_for')
    children = 'datasets'

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

    # View for the node
    view = View()


ExperimentTreeEditor = TreeEditor(
    nodes=[
        RootNode(),
        ExperimentNode(),
        MeasurementNode(),
        CorrectedDatasetsNode(),
        DatasetNode(),
    ],
    selected='selected',
    editable=False,
)
