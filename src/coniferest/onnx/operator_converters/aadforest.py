from onnxconverter_common.registration import register_converter


def get_default_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["n_targets"] = 1
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["target_treeids"] = []
    attrs["target_nodeids"] = []
    attrs["target_ids"] = []
    attrs["target_weights"] = []
    return attrs


def get_leaf_weight(tree_id, node_id, parents, model):
    evaluator = model.evaluator
    selector_id = evaluator.node_offsets[tree_id].astype(int) + node_id
    selector = evaluator.selectors[selector_id]
    value = selector["value"]
    # Generally, selector["leaf"] should be -1 for leafs,
    # but for AADForest it stores related weight index.
    leaf_id = selector["left"]
    weight = evaluator.weights[leaf_id]

    return weight * value


def add_node(attr_pairs, tree_id, node_id, feature_id, mode, value, true_child_id, false_child_id):
    attr_pairs["nodes_treeids"].append(tree_id)
    attr_pairs["nodes_nodeids"].append(node_id)
    attr_pairs["nodes_featureids"].append(feature_id)
    attr_pairs["nodes_modes"].append(mode)
    attr_pairs["nodes_values"].append(value)
    attr_pairs["nodes_truenodeids"].append(true_child_id)
    attr_pairs["nodes_falsenodeids"].append(false_child_id)


def add_leaf(attr_pairs, tree_id, node_id, mode, weight):
    add_node(attr_pairs, tree_id, node_id, 0, mode, 0.0, 0, 0)

    attr_pairs["target_treeids"].append(tree_id)
    attr_pairs["target_nodeids"].append(node_id)
    attr_pairs["target_ids"].append(0)
    attr_pairs["target_weights"].append(weight)


def add_tree_to_attribute_pairs(attr_pairs, tree, tree_id, model):
    parents = {}

    for i in range(tree.node_count):
        node_id = i

        if tree.children_left[i] > i or tree.children_right[i] > i:
            mode = "BRANCH_LEQ"
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
            parents[left_child_id] = i
            parents[right_child_id] = i

            add_node(attr_pairs, tree_id, node_id, feat_id, mode, threshold, left_child_id, right_child_id)
        else:
            mode = "LEAF"
            weight = get_leaf_weight(tree_id, node_id, parents, model)

            add_leaf(attr_pairs, tree_id, node_id, mode, weight)


def convert_aadforest(scope, operator, container):
    model = operator.raw_operator

    attr_pairs = get_default_attribute_pairs()

    for i, tree in enumerate(model.trees):
        add_tree_to_attribute_pairs(attr_pairs, tree, i, model)

    container.add_node(
        "TreeEnsembleRegressor",
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
        **attr_pairs,
    )


register_converter("AADForest", convert_aadforest)
