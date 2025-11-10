import re
import networkx as nx
from collections import OrderedDict
from gymnasium.spaces import *

dfa_text = '''
0 -> 0 [label="~g1 & ~g8"];
0 -> 1 [label="g8 & ~g1"];
0 -> 2 [label="g1 & ~g8"];
0 -> 3 [label="g1 & g8"];
1 -> 1 [label="~g1 & ~g7"];
1 -> 4 [label="g7 & ~g1"];
1 -> 3 [label="g1 & ~g7"];
1 -> 5 [label="g1 & g7"];
2 -> 2 [label="~g2 & ~g8"];
2 -> 3 [label="g8 & ~g2"];
2 -> 6 [label="g2 & ~g8"];
2 -> 7 [label="g2 & g8"];
3 -> 3 [label="~g2 & ~g7"];
3 -> 5 [label="g7 & ~g2"];
3 -> 7 [label="g2 & ~g7"];
3 -> 8 [label="g2 & g7"];
4 -> 4 [label="~g1 & ~g6"];
4 -> 9 [label="g6 & ~g1"];
4 -> 5 [label="g1 & ~g6"];
4 -> 10 [label="g1 & g6"];
5 -> 5 [label="~g2 & ~g6"];
5 -> 10 [label="g6 & ~g2"];
5 -> 8 [label="g2 & ~g6"];
5 -> 11 [label="g2 & g6"];
6 -> 6 [label="~g3 & ~g8"];
6 -> 7 [label="g8 & ~g3"];
6 -> 12 [label="g3 & ~g8"];
6 -> 13 [label="g3 & g8"];
7 -> 7 [label="~g3 & ~g7"];
7 -> 8 [label="g7 & ~g3"];
7 -> 13 [label="g3 & ~g7"];
7 -> 14 [label="g3 & g7"];
8 -> 8 [label="~g3 & ~g6"];
8 -> 11 [label="g6 & ~g3"];
8 -> 14 [label="g3 & ~g6"];
8 -> 15 [label="g3 & g6"];
9 -> 9 [label="~g1 & ~g5"];
9 -> 16 [label="g5 & ~g1"];
9 -> 10 [label="g1 & ~g5"];
9 -> 17 [label="g1 & g5"];
10 -> 10 [label="~g2 & ~g5"];
10 -> 17 [label="g5 & ~g2"];
10 -> 11 [label="g2 & ~g5"];
10 -> 18 [label="g2 & g5"];
11 -> 11 [label="~g3 & ~g5"];
11 -> 18 [label="g5 & ~g3"];
11 -> 15 [label="g3 & ~g5"];
11 -> 19 [label="g3 & g5"];
12 -> 12 [label="~g4 & ~g8"];
12 -> 13 [label="g8 & ~g4"];
12 -> 20 [label="g4 & ~g8"];
12 -> 21 [label="g4 & g8"];
13 -> 13 [label="~g4 & ~g7"];
13 -> 14 [label="g7 & ~g4"];
13 -> 21 [label="g4 & ~g7"];
13 -> 22 [label="g4 & g7"];
14 -> 14 [label="~g4 & ~g6"];
14 -> 15 [label="g6 & ~g4"];
14 -> 22 [label="g4 & ~g6"];
14 -> 23 [label="g4 & g6"];
15 -> 15 [label="~g4 & ~g5"];
15 -> 19 [label="g5 & ~g4"];
15 -> 23 [label="g4 & ~g5"];
15 -> 24 [label="g4 & g5"];
16 -> 16 [label="~g1 & ~g4"];
16 -> 25 [label="g4 & ~g1"];
16 -> 17 [label="g1 & ~g4"];
16 -> 26 [label="g1 & g4"];
17 -> 17 [label="~g2 & ~g4"];
17 -> 26 [label="g4 & ~g2"];
17 -> 18 [label="g2 & ~g4"];
17 -> 27 [label="g2 & g4"];
18 -> 18 [label="~g3 & ~g4"];
18 -> 27 [label="g4 & ~g3"];
18 -> 19 [label="g3 & ~g4"];
18 -> 28 [label="g3 & g4"];
19 -> 19 [label="~g4"];
19 -> 29 [label="g4"];
20 -> 20 [label="~g5 & ~g8"];
20 -> 21 [label="g8 & ~g5"];
20 -> 30 [label="g5 & ~g8"];
20 -> 31 [label="g5 & g8"];
21 -> 21 [label="~g5 & ~g7"];
21 -> 22 [label="g7 & ~g5"];
21 -> 31 [label="g5 & ~g7"];
21 -> 32 [label="g5 & g7"];
22 -> 22 [label="~g5 & ~g6"];
22 -> 23 [label="g6 & ~g5"];
22 -> 32 [label="g5 & ~g6"];
22 -> 33 [label="g5 & g6"];
23 -> 23 [label="~g5"];
23 -> 34 [label="g5"];
24 -> 24 [label="~g4 & ~g5"];
24 -> 34 [label="g5 & ~g4"];
24 -> 29 [label="g4 & ~g5"];
24 -> 35 [label="g4 & g5"];
25 -> 25 [label="~g1 & ~g3"];
25 -> 36 [label="g3 & ~g1"];
25 -> 26 [label="g1 & ~g3"];
25 -> 37 [label="g1 & g3"];
26 -> 26 [label="~g2 & ~g3"];
26 -> 37 [label="g3 & ~g2"];
26 -> 27 [label="g2 & ~g3"];
26 -> 38 [label="g2 & g3"];
27 -> 27 [label="~g3"];
27 -> 39 [label="g3"];
28 -> 28 [label="~g3 & ~g4"];
28 -> 29 [label="g4 & ~g3"];
28 -> 39 [label="g3 & ~g4"];
28 -> 40 [label="g3 & g4"];
29 -> 29 [label="~g3 & ~g5"];
29 -> 35 [label="g5 & ~g3"];
29 -> 40 [label="g3 & ~g5"];
29 -> 41 [label="g3 & g5"];
30 -> 30 [label="~g6 & ~g8"];
30 -> 31 [label="g8 & ~g6"];
30 -> 42 [label="g6 & ~g8"];
30 -> 43 [label="g6 & g8"];
31 -> 31 [label="~g6 & ~g7"];
31 -> 32 [label="g7 & ~g6"];
31 -> 43 [label="g6 & ~g7"];
31 -> 44 [label="g6 & g7"];
32 -> 32 [label="~g6"];
32 -> 45 [label="g6"];
33 -> 33 [label="~g5 & ~g6"];
33 -> 45 [label="g6 & ~g5"];
33 -> 34 [label="g5 & ~g6"];
33 -> 46 [label="g5 & g6"];
34 -> 34 [label="~g4 & ~g6"];
34 -> 46 [label="g6 & ~g4"];
34 -> 35 [label="g4 & ~g6"];
34 -> 47 [label="g4 & g6"];
35 -> 35 [label="~g3 & ~g6"];
35 -> 47 [label="g6 & ~g3"];
35 -> 41 [label="g3 & ~g6"];
35 -> 48 [label="g3 & g6"];
36 -> 36 [label="~g1 & ~g2"];
36 -> 49 [label="g2 & ~g1"];
36 -> 37 [label="g1 & ~g2"];
36 -> 50 [label="g1 & g2"];
37 -> 37 [label="~g2"];
37 -> 51 [label="g2"];
38 -> 38 [label="~g2 & ~g3"];
38 -> 39 [label="g3 & ~g2"];
38 -> 51 [label="g2 & ~g3"];
38 -> 52 [label="g2 & g3"];
39 -> 39 [label="~g2 & ~g4"];
39 -> 40 [label="g4 & ~g2"];
39 -> 52 [label="g2 & ~g4"];
39 -> 53 [label="g2 & g4"];
40 -> 40 [label="~g2 & ~g5"];
40 -> 41 [label="g5 & ~g2"];
40 -> 53 [label="g2 & ~g5"];
40 -> 54 [label="g2 & g5"];
41 -> 41 [label="~g2 & ~g6"];
41 -> 48 [label="g6 & ~g2"];
41 -> 54 [label="g2 & ~g6"];
41 -> 55 [label="g2 & g6"];
42 -> 42 [label="~g7 & ~g8"];
42 -> 43 [label="g8 & ~g7"];
42 -> 56 [label="g7 & ~g8"];
42 -> 57 [label="g7 & g8"];
43 -> 43 [label="~g7"];
43 -> 58 [label="g7"];
44 -> 44 [label="~g6 & ~g7"];
44 -> 58 [label="g7 & ~g6"];
44 -> 45 [label="g6 & ~g7"];
44 -> 59 [label="g6 & g7"];
45 -> 45 [label="~g5 & ~g7"];
45 -> 59 [label="g7 & ~g5"];
45 -> 46 [label="g5 & ~g7"];
45 -> 60 [label="g5 & g7"];
46 -> 46 [label="~g4 & ~g7"];
46 -> 60 [label="g7 & ~g4"];
46 -> 47 [label="g4 & ~g7"];
46 -> 61 [label="g4 & g7"];
47 -> 47 [label="~g3 & ~g7"];
47 -> 61 [label="g7 & ~g3"];
47 -> 48 [label="g3 & ~g7"];
47 -> 62 [label="g3 & g7"];
48 -> 48 [label="~g2 & ~g7"];
48 -> 62 [label="g7 & ~g2"];
48 -> 55 [label="g2 & ~g7"];
48 -> 63 [label="g2 & g7"];
49 -> 49 [label="~g1"];
49 -> 64 [label="g1"];
50 -> 50 [label="~g1 & ~g2"];
50 -> 51 [label="g2 & ~g1"];
50 -> 64 [label="g1"];
51 -> 51 [label="~g1 & ~g3"];
51 -> 52 [label="g3 & ~g1"];
51 -> 64 [label="g1"];
52 -> 52 [label="~g1 & ~g4"];
52 -> 53 [label="g4 & ~g1"];
52 -> 64 [label="g1"];
53 -> 53 [label="~g1 & ~g5"];
53 -> 54 [label="g5 & ~g1"];
53 -> 64 [label="g1"];
54 -> 54 [label="~g1 & ~g6"];
54 -> 55 [label="g6 & ~g1"];
54 -> 64 [label="g1"];
55 -> 55 [label="~g1 & ~g7"];
55 -> 63 [label="g7 & ~g1"];
55 -> 64 [label="g1"];
56 -> 56 [label="~g8"];
56 -> 64 [label="g8"];
57 -> 57 [label="~g7 & ~g8"];
57 -> 64 [label="g8"];
57 -> 58 [label="g7 & ~g8"];
58 -> 58 [label="~g6 & ~g8"];
58 -> 64 [label="g8"];
58 -> 59 [label="g6 & ~g8"];
59 -> 59 [label="~g5 & ~g8"];
59 -> 64 [label="g8"];
59 -> 60 [label="g5 & ~g8"];
60 -> 60 [label="~g4 & ~g8"];
60 -> 64 [label="g8"];
60 -> 61 [label="g4 & ~g8"];
61 -> 61 [label="~g3 & ~g8"];
61 -> 64 [label="g8"];
61 -> 62 [label="g3 & ~g8"];
62 -> 62 [label="~g2 & ~g8"];
62 -> 64 [label="g8"];
62 -> 63 [label="g2 & ~g8"];
63 -> 63 [label="~g1 & ~g8"];
63 -> 64 [label="g1 | g8"];
64 -> 64 [label="True"];'''

accepting_state = '64'

class DFATransformer:
    def __init__(self,):

        self.dfa = self.get_dfa(dfa_text)
        self.dfa_state = '0'
        self.accepting_state = accepting_state
        self.error_states = self.get_error_states()

    def get_error_states(self):
        self_loops = [node for node in self.dfa.nodes() if list(self.dfa.out_edges(node)) == [(node, node)]]
        self_loops.remove(self.accepting_state)
        return self_loops

    def get_dfa(self, text):
        G = nx.DiGraph()
        # 提取谓词
        predicates = self.extract_predicates(text)
        for predicate, start_state, end_state in predicates:
            G.add_node(start_state)
            G.add_node(end_state)
            G.add_edge(start_state, end_state, formula=predicate)
        return G

    def extract_predicates(self, text):
        # 匹配谓词的正则表达式，寻找 ^ 和 ) 之间的内容
        # 使用正则表达式匹配每一行

        pattern = r'(\d+) -> (\d+) \[label="([^"]+)"\];'
        matches = re.findall(pattern, text)
        # 将匹配结果转换为元组形式
        predicates = [(m[2], m[0], m[1]) for m in matches]
        return predicates

    def evaluate_logic_formula(self, props:Dict, formula):
        # 创建一个字典，将命题名称映射到它们的值
        '''props = {'p1': p1,'p2': p2,'p3': p3}'''

        # 替换公式中的命题名称为对应的布尔值
        # f = formula
        for var, value in props.items():
            formula = re.sub(r'\b' + re.escape(var) + r'\b', str(value), formula)
        formula = formula.replace('~', ' not ').replace('&', ' and ').replace('|', ' or ').replace('^', ' and ')
        # 评估公式
        try:
            result = eval(formula)
            return result
        except Exception as e:
            print(f"评估公式时出错: {e}")
            return None

    def reset(self):
        self.dfa_state = '0'

    # return terminate, if_success and if_failure
    def step(self, props):
        out_edges = self.dfa.out_edges(str(self.dfa_state), data=True)
        for edge in out_edges:
            if self.evaluate_logic_formula(props, formula=edge[2]['formula']):
                self.dfa_state = edge[1]
                break
        if self.dfa_state in self.error_states:
            return True, False, True, self.dfa_state
        if self.dfa_state == self.accepting_state:
            return True, True, False, self.dfa_state
        return False, False, False, self.dfa_state

