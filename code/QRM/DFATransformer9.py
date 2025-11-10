import re
import networkx as nx
from collections import OrderedDict
from gymnasium.spaces import *

dfa_text = '''
 0 -> 0 [label="~b & ~d"];
 0 -> 1 [label="d & ~b"];
 0 -> 2 [label="b & ~d"];
 0 -> 3 [label="b & d"];
 1 -> 1 [label="~c & ~e"];
 1 -> 7 [label="e & ~c"];
 1 -> 4 [label="c"];
 2 -> 2 [label="~a & ~c"];
 2 -> 7 [label="a & ~c"];
 2 -> 5 [label="c"];
 3 -> 3 [label="~a & ~c & ~e"];
 3 -> 1 [label="a & ~c & ~e"];
 3 -> 2 [label="e & ~a & ~c"];
 3 -> 7 [label="a & e & ~c"];
 3 -> 6 [label="c"];
 7 -> 7 [label="True"];
 4 -> 4 [label="~b & ~d"];
 4 -> 7 [label="d & ~b"];
 4 -> 8 [label="b"];
 5 -> 5 [label="~b & ~d"];
 5 -> 8 [label="d"];
 5 -> 7 [label="b & ~d"];
 6 -> 6 [label="~b & ~d"];
 6 -> 8 [label="b | d"];
 8 -> 8 [label="True"];'''

accepting_state = '8'

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

        for var, value in props.items():
            formula = re.sub(r'\b' + re.escape(var) + r'\b', str(value), formula)
        formula = formula.replace('~', 'not ').replace('&', 'and ').replace('|', 'or ')
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

