import re
import networkx as nx
from collections import OrderedDict
from gymnasium.spaces import *

dfa_text = '''
0 -> 0 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
0 -> 24 [label="(p2 & ~p1) | (p3 & ~p1) | (p4 & ~p1) | (p6 & ~p5) | (p7 & ~p5) | (p8 & ~p5)"];
0 -> 1 [label="p5 & ~p1 & ~p2 & ~p3 & ~p4"];
0 -> 2 [label="p1 & ~p5 & ~p6 & ~p7 & ~p8"];
0 -> 3 [label="p1 & p5"];
24 -> 24 [label="true"];
1 -> 1 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
1 -> 24 [label="(p2 & ~p1) | (p3 & ~p1) | (p4 & ~p1) | (p5 & ~p6) | (p7 & ~p6) | (p8 & ~p6)"];
1 -> 4 [label="p6 & ~p1 & ~p2 & ~p3 & ~p4"];
1 -> 3 [label="p1 & ~p5 & ~p6 & ~p7 & ~p8"];
1 -> 5 [label="p1 & p6"];
2 -> 2 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
2 -> 24 [label="(p1 & ~p2) | (p3 & ~p2) | (p4 & ~p2) | (p6 & ~p5) | (p7 & ~p5) | (p8 & ~p5)"];
2 -> 3 [label="p5 & ~p1 & ~p2 & ~p3 & ~p4"];
2 -> 6 [label="p2 & ~p5 & ~p6 & ~p7 & ~p8"];
2 -> 7 [label="p2 & p5"];
3 -> 3 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
3 -> 24 [label="(p1 & ~p2) | (p3 & ~p2) | (p4 & ~p2) | (p5 & ~p6) | (p7 & ~p6) | (p8 & ~p6)"];
3 -> 5 [label="p6 & ~p1 & ~p2 & ~p3 & ~p4"];
3 -> 7 [label="p2 & ~p5 & ~p6 & ~p7 & ~p8"];
3 -> 8 [label="p2 & p6"];
4 -> 4 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
4 -> 24 [label="(p2 & ~p1) | (p3 & ~p1) | (p4 & ~p1) | (p5 & ~p7) | (p6 & ~p7) | (p8 & ~p7)"];
4 -> 9 [label="p7 & ~p1 & ~p2 & ~p3 & ~p4"];
4 -> 5 [label="p1 & ~p5 & ~p6 & ~p7 & ~p8"];
4 -> 10 [label="p1 & p7"];
5 -> 5 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
5 -> 24 [label="(p1 & ~p2) | (p3 & ~p2) | (p4 & ~p2) | (p5 & ~p7) | (p6 & ~p7) | (p8 & ~p7)"];
5 -> 10 [label="p7 & ~p1 & ~p2 & ~p3 & ~p4"];
5 -> 8 [label="p2 & ~p5 & ~p6 & ~p7 & ~p8"];
5 -> 11 [label="p2 & p7"];
6 -> 6 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
6 -> 24 [label="(p1 & ~p3) | (p2 & ~p3) | (p4 & ~p3) | (p6 & ~p5) | (p7 & ~p5) | (p8 & ~p5)"];
6 -> 7 [label="p5 & ~p1 & ~p2 & ~p3 & ~p4"];
6 -> 12 [label="p3 & ~p5 & ~p6 & ~p7 & ~p8"];
6 -> 13 [label="p3 & p5"];
7 -> 7 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
7 -> 24 [label="(p1 & ~p3) | (p2 & ~p3) | (p4 & ~p3) | (p5 & ~p6) | (p7 & ~p6) | (p8 & ~p6)"];
7 -> 8 [label="p6 & ~p1 & ~p2 & ~p3 & ~p4"];
7 -> 13 [label="p3 & ~p5 & ~p6 & ~p7 & ~p8"];
7 -> 14 [label="p3 & p6"];
8 -> 8 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
8 -> 24 [label="(p1 & ~p3) | (p2 & ~p3) | (p4 & ~p3) | (p5 & ~p7) | (p6 & ~p7) | (p8 & ~p7)"];
8 -> 11 [label="p7 & ~p1 & ~p2 & ~p3 & ~p4"];
8 -> 14 [label="p3 & ~p5 & ~p6 & ~p7 & ~p8"];
8 -> 15 [label="p3 & p7"];
9 -> 9 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
9 -> 16 [label="p8 & ~p1 & ~p2 & ~p3 & ~p4"];
9 -> 24 [label="(p2 & ~p1) | (p3 & ~p1) | (p4 & ~p1) | (p5 & ~p8) | (p6 & ~p8) | (p7 & ~p8)"];
9 -> 10 [label="p1 & ~p5 & ~p6 & ~p7 & ~p8"];
9 -> 17 [label="p1 & p8"];
10 -> 10 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
10 -> 17 [label="p8 & ~p1 & ~p2 & ~p3 & ~p4"];
10 -> 24 [label="(p1 & ~p2) | (p3 & ~p2) | (p4 & ~p2) | (p5 & ~p8) | (p6 & ~p8) | (p7 & ~p8)"];
10 -> 11 [label="p2 & ~p5 & ~p6 & ~p7 & ~p8"];
10 -> 18 [label="p2 & p8"];
11 -> 11 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
11 -> 18 [label="p8 & ~p1 & ~p2 & ~p3 & ~p4"];
11 -> 24 [label="(p1 & ~p3) | (p2 & ~p3) | (p4 & ~p3) | (p5 & ~p8) | (p6 & ~p8) | (p7 & ~p8)"];
11 -> 15 [label="p3 & ~p5 & ~p6 & ~p7 & ~p8"];
11 -> 19 [label="p3 & p8"];
12 -> 12 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
12 -> 24 [label="(p1 & ~p4) | (p2 & ~p4) | (p3 & ~p4) | (p6 & ~p5) | (p7 & ~p5) | (p8 & ~p5)"];
12 -> 13 [label="p5 & ~p1 & ~p2 & ~p3 & ~p4"];
12 -> 20 [label="p4 & ~p5 & ~p6 & ~p7 & ~p8"];
12 -> 21 [label="p4 & p5"];
13 -> 13 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
13 -> 24 [label="(p1 & ~p4) | (p2 & ~p4) | (p3 & ~p4) | (p5 & ~p6) | (p7 & ~p6) | (p8 & ~p6)"];
13 -> 14 [label="p6 & ~p1 & ~p2 & ~p3 & ~p4"];
13 -> 21 [label="p4 & ~p5 & ~p6 & ~p7 & ~p8"];
13 -> 22 [label="p4 & p6"];
14 -> 14 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
14 -> 24 [label="(p1 & ~p4) | (p2 & ~p4) | (p3 & ~p4) | (p5 & ~p7) | (p6 & ~p7) | (p8 & ~p7)"];
14 -> 15 [label="p7 & ~p1 & ~p2 & ~p3 & ~p4"];
14 -> 22 [label="p4 & ~p5 & ~p6 & ~p7 & ~p8"];
14 -> 23 [label="p4 & p7"];
15 -> 15 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6 & ~p7 & ~p8"];
15 -> 19 [label="p8 & ~p1 & ~p2 & ~p3 & ~p4"];
15 -> 24 [label="(p1 & ~p4) | (p2 & ~p4) | (p3 & ~p4) | (p5 & ~p8) | (p6 & ~p8) | (p7 & ~p8)"];
15 -> 23 [label="p4 & ~p5 & ~p6 & ~p7 & ~p8"];
15 -> 25 [label="p4 & p8"];
16 -> 16 [label="~p1 & ~p2 & ~p3 & ~p4"];
16 -> 24 [label="~p1 & (p2 | p3 | p4)"];
16 -> 17 [label="p1"];
17 -> 17 [label="~p1 & ~p2 & ~p3 & ~p4"];
17 -> 24 [label="~p2 & (p1 | p3 | p4)"];
17 -> 18 [label="p2"];
18 -> 18 [label="~p1 & ~p2 & ~p3 & ~p4"];
18 -> 24 [label="~p3 & (p1 | p2 | p4)"];
18 -> 19 [label="p3"];
19 -> 19 [label="~p1 & ~p2 & ~p3 & ~p4"];
19 -> 25 [label="p4"];
19 -> 24 [label="~p4 & (p1 | p2 | p3)"];
20 -> 20 [label="~p5 & ~p6 & ~p7 & ~p8"];
20 -> 24 [label="~p5 & (p6 | p7 | p8)"];
20 -> 21 [label="p5"];
21 -> 21 [label="~p5 & ~p6 & ~p7 & ~p8"];
21 -> 24 [label="~p6 & (p5 | p7 | p8)"];
21 -> 22 [label="p6"];
22 -> 22 [label="~p5 & ~p6 & ~p7 & ~p8"];
22 -> 24 [label="~p7 & (p5 | p6 | p8)"];
22 -> 23 [label="p7"];
23 -> 23 [label="~p5 & ~p6 & ~p7 & ~p8"];
23 -> 25 [label="p8"];
23 -> 24 [label="~p8 & (p5 | p6 | p7)"];
25 -> 25 [label="true"];'''

accepting_state = '25'

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

