import re

# Define the mapping dictionary
mapping = {
    '1': '0',
    '2': '3',
    '3': '1',
    '4': '2',
    '5': '4',
    # 可以在这里添加更多的映射
}
'''for i in range(1, 66):
    mapping[str(i)] = str(i - 1)
print(mapping)'''
# Input graph definitions as a multi-line string
input_graph = """
 1 -> 1 [label="~c1 & ~c2 & ~c3"];
 1 -> 2 [label="c2 | c3"];
 1 -> 3 [label="c1 & ~c2 & ~c3"];
 2 -> 2 [label="True"];
 3 -> 2 [label="c3 | ~c1"];
 3 -> 3 [label="c1 & ~c2 & ~c3"];
 3 -> 4 [label="c1 & c2 & ~c3"];
 4 -> 2 [label="~c1 | ~c2"];
 4 -> 4 [label="c1 & c2 & ~c3"];
 4 -> 5 [label="c1 & c2 & c3"];
 5 -> 2 [label="~c1 | ~c2 | ~c3"];
 5 -> 5 [label="c1 & c2 & c3"];
"""

# Compile a regular expression pattern to match "x -> y [label="..."];"
pattern = re.compile(r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]')


# Function to replace node numbers based on the mapping
def replace_nodes(line, mapping, pattern):
    match = pattern.match(line.strip())
    if match:
        src, dst, label = match.groups()
        # Get the mapped values, if not found keep original
        new_src = mapping.get(src, src)
        new_dst = mapping.get(dst, dst)
        return f"{new_src} -> {new_dst} [label=\"{label}\"];"
    else:
        # If the line doesn't match the pattern, return it unchanged
        return line


# Process each line
output_lines = []
for line in input_graph.strip().split('\n'):
    new_line = replace_nodes(line, mapping, pattern)
    output_lines.append(new_line)

# Combine the transformed lines into a single string
output_graph = '\n'.join(output_lines)

# Print the transformed graph
print(output_graph)

