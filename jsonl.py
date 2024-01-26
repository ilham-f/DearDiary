import json

def read_jsonl_file(file_path):
    """
    Read a JSON Lines (JSONL) file and return a list of dictionaries.

    Parameters:
    - file_path (str): The path to the JSONL file.

    Returns:
    - data (list): A list of dictionaries representing each JSON object in the file.
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line}")
                print(e)

    return data

# Example usage:
file_path = 'tweet.jsonl'
jsonl_data = read_jsonl_file(file_path)

# Now, 'jsonl_data' contains a list of dictionaries, each representing a JSON object from the JSONL file.
