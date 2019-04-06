import json
import math

from functools import partial

from shared import get_csv_dataset_text

def get_counters(dataset, column_index):
    counters = {}
    length = len(dataset)
    for row in dataset:
        counter = counters.get(row[column_index], 0)
        counters[row[column_index]] = counter + 1

    return counters, length

def get_class_values(dataset, at_index):
    return set(row[at_index] for row in dataset)
    
def convert_counter_to_probability(counter):
    counters, length = counter
    return {
        key: count / length
    for key, count in counters.items()}

def find_probablities(dataset, headers, root_class_header):
    root_class_index = headers.index(root_class_header)
    root_counter = get_counters(dataset, root_class_index)
    probabilities = {
        root_class_header: convert_counter_to_probability(root_counter)
    }
    for label in get_class_values(dataset, root_class_index):
        probabilities[label] = {}
        for i, header in enumerate(headers):
            if i == root_class_index:
                continue
            counter = get_counters([
                row
                for row in dataset 
                if row[root_class_index] == label 
            ], i)
            probabilities[label][header] = convert_counter_to_probability(counter)

    return probabilities

def calculate_choices(probabilities, conditions, root_class_header, is_map=True):
    local_probabilities = probabilities.copy()
    root_probabilities = local_probabilities.pop(root_class_header)
    options = {}
    for label, probability in root_probabilities.items():
        final_probability = probability if is_map else 1
        values = local_probabilities[label]
        for key, condition in conditions.items():
            if key in values:
                final_probability *= values[key][condition]

        options[label] = final_probability

    return options

calculate_map = partial(calculate_choices, is_map=True)
calculate_ml = partial(calculate_choices, is_map=False)

def make_choice(values):
    return max(values.items(), key=lambda x: x[1])[0]

def make_and_print_choice(values, root_class_header, choice_type='Unknown'):
    choice = make_choice(values)
    print(f'{choice_type} {root_class_header}: {choice}')
    return choice


if __name__ == "__main__":
    import json
    headers, *dataset = get_csv_dataset_text('golf-play')
    root_class_header = headers[-1]
    probabilities = find_probablities(dataset, headers, root_class_header)
    # Test conditions
    # Outlook = Sunny, Temperature = Cool , 
    # Humidity = High, Wind = Strong
    current_conditions = {
        "Outlook": "Sunny",
        "Temp": "Cool",
        "Humidity": "High",
        "Wind": "Strong"
    }
    map_values = calculate_map(probabilities, current_conditions, root_class_header)
    ml_values = calculate_ml(probabilities, current_conditions, root_class_header)
    print("Probabilities", json.dumps(probabilities, indent=True))
    print("Choices", json.dumps(current_conditions, indent=True))
    print("MAP Probabilities", map_values)
    print("ML Probabilities", ml_values)
    make_and_print_choice(map_values, root_class_header, 'MAP')
    make_and_print_choice(ml_values, root_class_header, 'ML')