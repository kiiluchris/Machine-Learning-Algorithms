import math
from shared import get_csv_dataset_text



def get_class_values(dataset, at_index):
    return set(row[at_index] for row in dataset)

def get_counters(dataset, column_index):
    counters = {}
    length = len(dataset)
    for row in dataset:
        counter = counters.get(row[column_index], 0)
        counters[row[column_index]] = counter + 1

    return counters, length

def calculate_entropy(dataset, counters, length):
    return sum(-(counter/length) * math.log2(counter/length) for counter in counters.values())

def calculate_information_gain(dataset, entropy, column_index, label, class_index):
    for row in dataset:
        row[column_index]
    return entropy - sum()

def gen_tree(orig_dataset, orig_headers, root_class_index=-1):
    dataset = orig_dataset.copy()
    headers = orig_headers[:]
    tree = {}
    label_counters, label_length = get_counters(dataset, root_class_index)
    entropy = calculate_entropy(dataset, label_counters, label_length)
    tree = {
        "entropy": entropy,
        "children": {}
    }
    for i, header in enumerate(headers):
        if i != root_class_index:
            tree["children"][header] = {
                "values": {}
            }
            for label in get_class_values(dataset, i):
                subset = [row for row in dataset if row[i] == label]
                counters, length = get_counters(subset, root_class_index)
                dependant_entropy = calculate_entropy(subset, counters, length)
                probability = length / label_length
                tree["children"][header]["values"][label] = {
                    "entropy": dependant_entropy,
                    "probablity": probability,
                    "neg_prod": -probability * dependant_entropy
                }
            tree["children"][header]["information_gain"] = entropy + sum(node["neg_prod"] for node in tree["children"][header]["values"].values())

    return tree

def get_max_attribute(tree):
    max_header, max_gain = '', -1
    for header, data in tree["children"].items():
        if max_gain < data['information_gain']:
            max_gain = data['information_gain']
            max_header = header
    return max_header

def all_vals_same(xs):
    arr = list(xs)
    x = arr[0]
    for y in arr[1:]:
        if y != x: 
            return False
    return True

def _make_decision(dataset, headers, root_class_index):
    tree = gen_tree(dataset, headers, root_class_index)
    max_header = get_max_attribute(tree)
    max_header_index = headers.index(max_header)
    classes = get_class_values(dataset, max_header_index)
    final_tree = {
        max_header: {}
    }
    for label in classes:
        subset = [row for row in dataset if row[max_header_index] == label]
        counters, _ = get_counters(subset, root_class_index)
        counter_keys = list(counters.keys())
        if len(counter_keys) == 1:
            final_tree[max_header][label] = counter_keys[0]
        # elif all_vals_same(counters.values()):
        #     final_tree[max_header][label] = 'Unknown'
        else:
            final_tree[max_header][label] = _make_decision(subset, headers, root_class_index)

    return final_tree

def make_decision(dataset, headers, root_class_header):
    root_class_index = headers.index(root_class_header)
    return {
        headers[root_class_index]: _make_decision(dataset, headers, root_class_index)
    }



headers, *dataset = get_csv_dataset_text('golf-play')
tree = make_decision(dataset, headers, headers[-1])
import json
print(json.dumps(tree, indent=True))