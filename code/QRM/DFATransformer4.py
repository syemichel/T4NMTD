import re
import networkx as nx
from collections import OrderedDict
from gymnasium.spaces import *

dfa_text = '''
0 -> 0 [label="~g1 & ~g6"];
0 -> 1 [label="g6 & ~g1"];
0 -> 2 [label="g1 & ~g6"];
0 -> 3 [label="g1 & g6"];
1 -> 1 [label="~g1 & ~g5"];
1 -> 4 [label="g5 & ~g1"];
1 -> 3 [label="g1 & ~g5"];
1 -> 5 [label="g1 & g5"];
2 -> 2 [label="~g2 & ~g6"];
2 -> 3 [label="g6 & ~g2"];
2 -> 6 [label="g2 & ~g6"];
2 -> 7 [label="g2 & g6"];
3 -> 3 [label="~g2 & ~g5"];
3 -> 5 [label="g5 & ~g2"];
3 -> 7 [label="g2 & ~g5"];
3 -> 8 [label="g2 & g5"];
4 -> 4 [label="~g1 & ~g4"];
4 -> 9 [label="g4 & ~g1"];
4 -> 5 [label="g1 & ~g4"];
4 -> 10 [label="g1 & g4"];
5 -> 5 [label="~g2 & ~g4"];
5 -> 10 [label="g4 & ~g2"];
5 -> 8 [label="g2 & ~g4"];
5 -> 11 [label="g2 & g4"];
6 -> 6 [label="~g3 & ~g6"];
6 -> 7 [label="g6 & ~g3"];
6 -> 12 [label="g3 & ~g6"];
6 -> 13 [label="g3 & g6"];
7 -> 7 [label="~g3 & ~g5"];
7 -> 8 [label="g5 & ~g3"];
7 -> 13 [label="g3 & ~g5"];
7 -> 14 [label="g3 & g5"];
8 -> 8 [label="~g3 & ~g4"];
8 -> 11 [label="g4 & ~g3"];
8 -> 14 [label="g3 & ~g4"];
8 -> 15 [label="g3 & g4"];
9 -> 9 [label="~g1 & ~g3"];
9 -> 16 [label="g3 & ~g1"];
9 -> 10 [label="g1 & ~g3"];
9 -> 17 [label="g1 & g3"];
10 -> 10 [label="~g2 & ~g3"];
10 -> 17 [label="g3 & ~g2"];
10 -> 11 [label="g2 & ~g3"];
10 -> 18 [label="g2 & g3"];
11 -> 11 [label="~g3"];
11 -> 19 [label="g3"];
12 -> 12 [label="~g4 & ~g6"];
12 -> 13 [label="g6 & ~g4"];
12 -> 20 [label="g4 & ~g6"];
12 -> 21 [label="g4 & g6"];
13 -> 13 [label="~g4 & ~g5"];
13 -> 14 [label="g5 & ~g4"];
13 -> 21 [label="g4 & ~g5"];
13 -> 22 [label="g4 & g5"];
14 -> 14 [label="~g4"];
14 -> 23 [label="g4"];
15 -> 15 [label="~g3 & ~g4"];
15 -> 23 [label="g4 & ~g3"];
15 -> 19 [label="g3 & ~g4"];
15 -> 24 [label="g3 & g4"];
16 -> 16 [label="~g1 & ~g2"];
16 -> 25 [label="g2 & ~g1"];
16 -> 17 [label="g1 & ~g2"];
16 -> 26 [label="g1 & g2"];
17 -> 17 [label="~g2"];
17 -> 27 [label="g2"];
18 -> 18 [label="~g2 & ~g3"];
18 -> 19 [label="g3 & ~g2"];
18 -> 27 [label="g2 & ~g3"];
18 -> 28 [label="g2 & g3"];
19 -> 19 [label="~g2 & ~g4"];
19 -> 24 [label="g4 & ~g2"];
19 -> 28 [label="g2 & ~g4"];
19 -> 29 [label="g2 & g4"];
20 -> 20 [label="~g5 & ~g6"];
20 -> 21 [label="g6 & ~g5"];
20 -> 30 [label="g5 & ~g6"];
20 -> 31 [label="g5 & g6"];
21 -> 21 [label="~g5"];
21 -> 32 [label="g5"];
22 -> 22 [label="~g4 & ~g5"];
22 -> 32 [label="g5 & ~g4"];
22 -> 23 [label="g4 & ~g5"];
22 -> 33 [label="g4 & g5"];
23 -> 23 [label="~g3 & ~g5"];
23 -> 33 [label="g5 & ~g3"];
23 -> 24 [label="g3 & ~g5"];
23 -> 34 [label="g3 & g5"];
24 -> 24 [label="~g2 & ~g5"];
24 -> 34 [label="g5 & ~g2"];
24 -> 29 [label="g2 & ~g5"];
24 -> 35 [label="g2 & g5"];
25 -> 25 [label="~g1"];
25 -> 36 [label="g1"];
26 -> 26 [label="~g1 & ~g2"];
26 -> 27 [label="g2 & ~g1"];
26 -> 36 [label="g1"];
27 -> 27 [label="~g1 & ~g3"];
27 -> 28 [label="g3 & ~g1"];
27 -> 36 [label="g1"];
28 -> 28 [label="~g1 & ~g4"];
28 -> 29 [label="g4 & ~g1"];
28 -> 36 [label="g1"];
29 -> 29 [label="~g1 & ~g5"];
29 -> 35 [label="g5 & ~g1"];
29 -> 36 [label="g1"];
30 -> 30 [label="~g6"];
30 -> 36 [label="g6"];
31 -> 31 [label="~g5 & ~g6"];
31 -> 36 [label="g6"];
31 -> 32 [label="g5 & ~g6"];
32 -> 32 [label="~g4 & ~g6"];
32 -> 36 [label="g6"];
32 -> 33 [label="g4 & ~g6"];
33 -> 33 [label="~g3 & ~g6"];
33 -> 36 [label="g6"];
33 -> 34 [label="g3 & ~g6"];
34 -> 34 [label="~g2 & ~g6"];
34 -> 36 [label="g6"];
34 -> 35 [label="g2 & ~g6"];
35 -> 35 [label="~g1 & ~g6"];
35 -> 36 [label="g1 | g6"];
36 -> 36 [label="True"];'''

accepting_state = '36'

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

