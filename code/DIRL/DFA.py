import re
import networkx as nx
from collections import OrderedDict
from gymnasium.spaces import *

class OrderedSet:
    def __init__(self):
        self._data = OrderedDict()

    def add(self, value):
        self._data[value] = None

    def __iter__(self):
        return iter(self._data.keys())

    def __contains__(self, value):
        return value in self._data

    def __len__(self):
        return len(self._data)

def extract_predicates(text):
    # 匹配谓词的正则表达式，寻找 ^ 和 ) 之间的内容
    pattern = re.compile(r'if\s*\(ds\s*==\s*(@q\d+)\s*\^\s*\((.*?)\)\s*\)\s*then\s*(@q\d+)')
    matches = pattern.findall(text)
    predicates = [(match[1], match[0], match[2]) for match in matches]
    return predicates

def get_dfa(text):
    G = nx.DiGraph()
    # 提取谓词
    predicates = extract_predicates(text)

    # 使用集合来存储唯一的谓词
    unique_predicates = OrderedSet()
    error_states = ['@q2']
    # 输出结果，排除自循环边的谓词
    for predicate, start_state, end_state in predicates:
        G.add_node(start_state)
        G.add_node(end_state)
        G.add_edge(start_state, end_state, formula=predicate)
    return G

class DFATransformer:
    def __init__(self, dfa_text):
        self.dfa = get_dfa(dfa_text)
        self.dfa_state = '@q1'
        self.accepting_state = "@q" + str(self.dfa.number_of_nodes())
        self.error_state = "@q2"

    def reset(self):
        self.dfa_state = '@q1'

    def evaluate_logic_formula(self, props:Dict, formula):
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

    # return terminate, if_success and if_failure
    def step(self, props):
        out_edges = self.dfa.out_edges(str(self.dfa_state), data=True)
        for edge in out_edges:
            if self.evaluate_logic_formula(props, formula=edge[2]['formula']):
                self.dfa_state = edge[1]
                break
        if self.dfa_state == self.error_state:
            return True, False, True
        if self.dfa_state == self.accepting_state:
            return True, True, False
        return False, False, False

def extract_true_predicates(proposition1, proposition2):
    # 匹配未被否定的原子命题，使用负向回顾确保前面没有~符号
    positive1 = re.findall(r'(?<!~)p\d+', proposition1)
    positive2 = re.findall(r'(?<!~)p\d+', proposition2)
    positive = list(set(positive1) - set(positive2))
    return positive