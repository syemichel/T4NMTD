import re
import networkx as nx
from collections import OrderedDict
from gymnasium.spaces import *

dfa_text = '''
0 -> 0 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
0 -> 15 [label="(p2 & ~p1) | (p3 & ~p1) | (p5 & ~p4) | (p6 & ~p4)"];
0 -> 1 [label="p4 & ~p1 & ~p2 & ~p3"];
0 -> 2 [label="p1 & ~p4 & ~p5 & ~p6"];
0 -> 3 [label="p1 & p4"];
15 -> 15 [label="True"];
1 -> 1 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
1 -> 15 [label="(p2 & ~p1) | (p3 & ~p1) | (p4 & ~p5) | (p6 & ~p5)"];
1 -> 4 [label="p5 & ~p1 & ~p2 & ~p3"];
1 -> 3 [label="p1 & ~p4 & ~p5 & ~p6"];
1 -> 5 [label="p1 & p5"];
2 -> 2 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
2 -> 15 [label="(p1 & ~p2) | (p3 & ~p2) | (p5 & ~p4) | (p6 & ~p4)"];
2 -> 3 [label="p4 & ~p1 & ~p2 & ~p3"];
2 -> 6 [label="p2 & ~p4 & ~p5 & ~p6"];
2 -> 7 [label="p2 & p4"];
3 -> 3 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
3 -> 15 [label="(p1 & ~p2) | (p3 & ~p2) | (p4 & ~p5) | (p6 & ~p5)"];
3 -> 5 [label="p5 & ~p1 & ~p2 & ~p3"];
3 -> 7 [label="p2 & ~p4 & ~p5 & ~p6"];
3 -> 8 [label="p2 & p5"];
4 -> 4 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
4 -> 9 [label="p6 & ~p1 & ~p2 & ~p3"];
4 -> 15 [label="(p2 & ~p1) | (p3 & ~p1) | (p4 & ~p6) | (p5 & ~p6)"];
4 -> 5 [label="p1 & ~p4 & ~p5 & ~p6"];
4 -> 10 [label="p1 & p6"];
5 -> 5 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
5 -> 10 [label="p6 & ~p1 & ~p2 & ~p3"];
5 -> 15 [label="(p1 & ~p2) | (p3 & ~p2) | (p4 & ~p6) | (p5 & ~p6)"];
5 -> 8 [label="p2 & ~p4 & ~p5 & ~p6"];
5 -> 11 [label="p2 & p6"];
6 -> 6 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
6 -> 15 [label="(p1 & ~p3) | (p2 & ~p3) | (p5 & ~p4) | (p6 & ~p4)"];
6 -> 7 [label="p4 & ~p1 & ~p2 & ~p3"];
6 -> 12 [label="p3 & ~p4 & ~p5 & ~p6"];
6 -> 13 [label="p3 & p4"];
7 -> 7 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
7 -> 15 [label="(p1 & ~p3) | (p2 & ~p3) | (p4 & ~p5) | (p6 & ~p5)"];
7 -> 8 [label="p5 & ~p1 & ~p2 & ~p3"];
7 -> 13 [label="p3 & ~p4 & ~p5 & ~p6"];
7 -> 14 [label="p3 & p5"];
8 -> 8 [label="~p1 & ~p2 & ~p3 & ~p4 & ~p5 & ~p6"];
8 -> 11 [label="p6 & ~p1 & ~p2 & ~p3"];
8 -> 15 [label="(p1 & ~p3) | (p2 & ~p3) | (p4 & ~p6) | (p5 & ~p6)"];
8 -> 14 [label="p3 & ~p4 & ~p5 & ~p6"];
8 -> 16 [label="p3 & p6"];
9 -> 9 [label="~p1 & ~p2 & ~p3"];
9 -> 15 [label="~p1 & (p2 | p3)"];
9 -> 10 [label="p1"];
10 -> 10 [label="~p1 & ~p2 & ~p3"];
10 -> 15 [label="~p2 & (p1 | p3)"];
10 -> 11 [label="p2"];
11 -> 11 [label="~p1 & ~p2 & ~p3"];
11 -> 16 [label="p3"];
11 -> 15 [label="~p3 & (p1 | p2)"];
12 -> 12 [label="~p4 & ~p5 & ~p6"];
12 -> 15 [label="~p4 & (p5 | p6)"];
12 -> 13 [label="p4"];
13 -> 13 [label="~p4 & ~p5 & ~p6"];
13 -> 15 [label="~p5 & (p4 | p6)"];
13 -> 14 [label="p5"];
14 -> 14 [label="~p4 & ~p5 & ~p6"];
14 -> 16 [label="p6"];
14 -> 15 [label="~p6 & (p4 | p5)"];
16 -> 16 [label="True"];'''

accepting_state = '16'

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

