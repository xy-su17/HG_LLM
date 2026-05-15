# coding=UTF-8
import argparse
import torch
import json
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import copy
import warnings
import pandas as pd
from utils import my_tokenizer, check

node_type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}

type_one_hot = np.eye(len(node_type_map))

edgeType_full = { 
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}

ver_edge_type = { 
    '1': 'IS_AST_PARENT',
    '2': 'IS_CLASS_OF',
    '3': 'FLOWS_TO',
    '4': 'EDF',
    '5': 'USE',
    '6': 'REACHES',
    '7': 'CONTROLS',
    '8': 'DECLARES',
    '9': 'DOM',
    '10': 'POST_DOM',
    '11': 'IS_FUNCTION_OF_AST',
    '12': 'IS_FUNCTION_OF_CFG'
}

edgeType_reduced = {
    'IS_AST_PARENT': 1,
    'FLOWS_TO': 2,
    'REACHES': 3,
    'NSC': 4,
    'Self_loop': 5,
    'HYPER_CALL': 6,  
    'HYPER_LOOP': 7,  
    'HYPER_IF': 8,  
    'HYPER_SWITCH': 9,  
    'HYPER_RETURN': 10,  
    'HYPER_PARAM': 11,  
    'HYPER_DATAFLOW': 12, 
    'HYPER_CONTROL': 13 
}

ver_edge_type_reduced = {
    '1': 'IS_AST_PARENT',
    '2': 'FLOWS_TO',
    '3': 'REACHES',
    '4': 'NSC',
    '5': 'Self_loop',
    '6': 'HYPER_CALL',
    '7': 'HYPER_LOOP',
    '8': 'HYPER_IF',
    '9': 'HYPER_SWITCH',
    '10': 'HYPER_RETURN',
    '11': 'HYPER_PARAM',
    '12': 'HYPER_DATAFLOW',
    '13': 'HYPER_CONTROL'
}

HYPER_EDGE_MIN_NODES = {
    'HYPER_CALL': 3,  
    'HYPER_LOOP': 2, 
    'HYPER_IF': 3,  
    'HYPER_SWITCH': 3,  
    'HYPER_RETURN': 2, 
    'HYPER_PARAM': 2,  
    'HYPER_DATAFLOW': 3,  
    'HYPER_CONTROL': 3 
}

allowed_edge_types_reduced = {
    'IS_AST_PARENT': 'black',
    'FLOWS_TO': 'blue',
    'REACHES': 'red',
    'NSC': 'red'
}

color_list = [
    'black',  # 黑色
    'blue',  # 蓝色
    'red',  # 红色
    'green',  # 绿色
    'purple',  # 紫色
    'orange',  # 橙色
    'yellow',  # 黄色
    'pink',  # 粉色
    'cyan',  # 青色
    'magenta',  # 洋红色
    'brown',  # 棕色
    'gray',  # 灰色
    'lime',  # 酸橙色
    'navy',  # 海军蓝
    'teal',  # 蓝绿色
    'maroon',  # 栗色
    'olive',  # 橄榄色
    'silver',  # 银色
    'gold',  # 金色
    'violet'  # 紫罗兰色
]

edge_type_ast = {
    'IS_AST_PARENT': 'black',
    'FLOWS_TO': 'blue',
    'REACHES': 'red',
    'NSC': 'green'
}

allowed_edge_types = {
    'FLOWS_TO': 'blue',  
    'REACHES': 'black',  
    'CONTROLS': 'black',  
    'DOM': 'red', 
    'POST_DOM': 'red',  
}

allowed_edge_types_full = {
    'IS_AST_PARENT': 'black',
    'IS_CLASS_OF': 'purple',
    'FLOWS_TO': 'blue',
    'DEF': 'green',
    'USE': 'green',
    'REACHES': 'black',
    'CONTROLS': 'black',
    'DECLARES': 'green',  
    'DOM': 'red',
    'POST_DOM': 'red',
    'IS_FUNCTION_OF_AST': 'green',
    'IS_FUNCTION_OF_CFG': 'green'
}

allowed_edge_types_dom = {
    'DOM': 'black',
    'POST_DOM': 'red'
}

allowed_edge_types_reach = {
    'REACHES': 'black'
}

allowed_edge_types_control = {
    'CONTROLS': 'black'
}

allowed_edge_types_def_use = {
    'DEF': 'black',
    # 'USE' : 'red'
}

allowed_edge_types_ast = {
    'IS_AST_PARENT': 'black',
    # 'USE' : 'red'
}


def checkVul(cFile):
    with open(cFile, 'r') as f:
        fileString = f.read()
        return (1 if "BUFWRITE_COND_UNSAFE" in fileString or "BUFWRITE_TAUT_UNSAFE" in fileString else 0)


warnings.filterwarnings('ignore')

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def build_ast(starts, edges, ast_edges):
    if len(starts) == 0:
        return
    new_starts = []
    for i in starts:
        ast = {}
        ast['start'] = i
        ast['end'] = []
        for edge in edges:
            if edge['start'].strip() == i and edge['type'].strip() == 'IS_AST_PARENT':
                ast['end'].append(edge['end'].strip())
                new_starts.append(edge['end'].strip())
        if len(ast['end']) > 0:
            ast_edges.append(ast)
    build_ast(new_starts, edges, ast_edges)


def get_nodes_by_key(nodes, key):
    for node in nodes:
        if node['key'].strip() == key:
            return node
    return None


def check_def(nodes, edges):
    defed = []
    for e in edges:
        if e['type'] == 'DEF':
            defed.append(e['end'])

def combine(x_node, y_node): 
    x_type = x_node['type'].strip()
    y_type = y_node['type'].strip()
    if x_type == 'ExpressionStatement': 
        if y_type == 'AssignmentExpression': 
            return True
        if y_type == 'UnaryExpression' and x_node['code'].strip() == y_node['code'].strip(): 
            return True
        if y_type == 'PostIncDecOperationExpression' and x_node['code'].strip() == y_node['code'].strip():
            return True
        if y_type == 'CallExpression':
            return True

    if x_type == 'IdentifierDeclStatement' and y_type == 'IdentifierDecl':  
        return True

    if x_type == 'CallExpression' and y_type == 'ArgumentList': 
        return True
    if x_type == 'Callee' and y_type == 'Identifier':  
        return True
    if x_type == 'Argument' and x_node['code'].strip() == y_node['code'].strip():  
        return True
    if x_type == 'Condition' and x_node['code'].strip() == y_node['code'].strip():  
        return True
    if x_type == 'ForInit':  
        if x_node['code'].strip() == y_node['code'].strip():
            return True
        if y_type == 'AssignmentExpression':
            return True

    return False


def check_dup_node(duplist, node_key): 
    for sublist in duplist:
        head = sublist[0]
        if node_key in sublist:
            return head
    return None


def ast_prune(nodes, ast_edges, ast_id):
    ast_new_edges = []
    duplist = []
    first_flag = True
    for item in ast_edges:
        repeat = []
        x = item['start']
        x_node = get_nodes_by_key(nodes, x)
        x_node_key = x_node['key'].strip()
        repeat.append(x_node_key)
        edges = {}
        p_node = x_node_key
        t_node = copy.deepcopy(item['end'])
        for y in item['end']:
            y_node = get_nodes_by_key(nodes, y)
            y_node_key = y_node['key'].strip()
            if combine(x_node, y_node) == True: 
                t_node.remove(y_node_key)  
                flag = False
                for sublist in duplist:  
                    if x_node_key in sublist:
                        sublist.append(y_node_key)  
                        p_node = sublist[0]  
                        flag = True
                        break
                if not flag:
                    repeat.append(y_node_key)  
            else:
                head = check_dup_node(duplist, x_node_key)
                if head != None:
                    p_node = head
        if len(repeat) > 1:
            duplist.append(repeat)
        edges = {}
        edges['start'] = p_node
        edges['end'] = t_node
        if first_flag == True: 
            first_flag = False
            edges['start'] = x_node_key
            ast_new_edges.append(edges)
        elif p_node == x_node_key and len(t_node) > 0: 
            ast_new_edges.append(edges)
        elif p_node != x_node_key:  
            find = False
            for i in ast_new_edges:
                if i['start'] == p_node:
                    find = True
                    i['end'] += t_node 
                    break
            if find == False:
                ast_new_edges.append(edges)
    return ast_new_edges, duplist


def edge_ver(index_map, edges):
    new_edges = []
    for e in edges:
        start, eType, end = e
        start = index_map[start]
        end = index_map[end]
        new_edge = [start, eType, end]
        new_edges.append(new_edge)
    return new_edges


def count_nodes_edges(edges):
    nodes = set()
    edges_num = 0
    for item in edges:
        nodes.add(item['start'])
        edges_num += len(item['end'])
        for y in item['end']:
            nodes.add(y)
    return len(nodes), edges_num


def get_ncs_edges(all_ast_edges, nsc_type, nodes):
    nsc_edges = list()
    first = True  
    par_sent = []
    tmp_sent = copy.deepcopy(all_ast_edges)
    for ast_sent in all_ast_edges:
        if get_nodes_by_key(nodes, ast_sent[0]['start'])['type'] == 'Parameter':
            par_sent.append(ast_sent)
            tmp_sent.remove(ast_sent)
    all_ast_edges = par_sent + tmp_sent

    for ast_sent in all_ast_edges:
        s_nodes = []
        t_nodes = []
        for ast in ast_sent:
            s_node = ast['start']
            s_nodes.append(s_node)
            t_nodes = t_nodes + ast['end']
        nsc_nodes = []
        if s_nodes == t_nodes and len(s_nodes) == 1:  # break; continue; return;
            nsc_nodes = t_nodes
        if len(s_nodes) == 1 and len(t_nodes) == 0:  # if(a) {start:'a', end:[]}
            nsc_nodes = s_nodes
        else:
            for node in t_nodes:
                if node not in s_nodes:
                    nsc_nodes.append(int(node))
        nsc_nodes.sort(reverse=False)
        idx = 0
        if first:
            first = False
            idx = 1
            s = nsc_nodes[0]
        for i in range(idx, len(nsc_nodes)):
            edge = [str(s), nsc_type, str(nsc_nodes[i])]
            nsc_edges.append(edge)
            s = nsc_nodes[i]
    return nsc_edges


def get_combine_ncs_edges(all_ast_edges, nsc_type, nodes, same_var):
    nsc_edges = list()
    first = True  
    par_sent = []
    tmp_sent = copy.deepcopy(all_ast_edges)
    for ast_sent in all_ast_edges:  
        if get_nodes_by_key(nodes, ast_sent[0]['start'])['type'] == 'Parameter': 
            par_sent.append(ast_sent)
            tmp_sent.remove(ast_sent)
    all_ast_edges = par_sent + tmp_sent 
    record_nodes = set()
    for ast_sent in all_ast_edges:
        s_nodes = []
        t_nodes = []
        for ast in ast_sent:
            s_node = ast['start']
            s_nodes.append(s_node)
            t_nodes = t_nodes + ast['end']
        nsc_nodes = []
        if s_nodes == t_nodes and len(s_nodes) == 1:  
            nsc_nodes = t_nodes
        if len(s_nodes) == 1 and len(t_nodes) == 0:  
            nsc_nodes = s_nodes
        else:  
            for node in t_nodes:
                if node not in s_nodes:
                    nsc_nodes.append(int(node))

        nsc_nodes.sort(reverse=False)
        nsc_nodes = [str(i) for i in nsc_nodes]

        combine_nsc_nodes = []
        for i, node in enumerate(nsc_nodes): 
            tmp = check_dup_node(same_var, node)  
            if tmp == None:
                raise (Exception("None Error!"))
            if tmp in record_nodes:  
                continue
            record_nodes.add(tmp)
            combine_nsc_nodes.append(tmp)
        idx = 0
        if first:  
            first = False
            idx = 1
            s = combine_nsc_nodes[0]  
        for i in range(idx, len(combine_nsc_nodes)):  
            edge = [str(s), nsc_type, str(combine_nsc_nodes[i])]
            nsc_edges.append(edge)
            s = combine_nsc_nodes[i]
    return nsc_edges


def spe_sent(n_type, n_code):
    if n_type == 'BreakStatement' and n_code == ['break', ';']:  
        return True
    elif n_type == 'ContinueStatement' and n_code == ['continue', ';']:  
        return True
    elif n_type == 'ReturnStatement' and n_code == ['return', ';']:  
        return True
    elif n_type == 'InfiniteForNode' and n_code == ['true']: 
        return True
    elif n_type == 'Label' and 'case' in n_code: 
        return True
    return False


def get_same_var(all_edges, nodes):  
    all_nodes = set()
    for ast in all_edges:
        for item in ast:
            all_nodes.add(item['start'])
            all_nodes.update(item['end']) 

    code_to_keys = {}
    other_key = []
    for node_key in all_nodes:
        node = get_nodes_by_key(nodes, node_key)

        node_type = node['type'].strip()
        if node_type not in ['Argument', 'Identifier']:
            other_key.append([node['key']])
            continue

        node_code = node['code'].strip()
        node_key_str = node['key'].strip()

        if node_code not in code_to_keys:
            code_to_keys[node_code] = []
        code_to_keys[node_code].append(node_key_str)

    sorted_same_var = []
    for code, keys in code_to_keys.items():
        if len(keys) > 1:  
            sorted_same_var.append(sorted(keys))
        else:
            sorted_same_var.append(keys)
    return sorted_same_var + other_key


def var_combine(ast_edges, same_var):  
    ast_combine_edges = copy.deepcopy(ast_edges)
    for ast_sent in ast_combine_edges:
        if len(ast_sent) == 1 and len(ast_sent[0]['end']) == 1 and ast_sent[0]['start'] == ast_sent[0]['end'][0]:  # 如果只有一个边且一个结束节点，且开始结束节点相同
            continue
        for item in ast_sent:
            heads = check_dup_node(same_var, item['start'])
            if heads != None:
                item['start'] = heads
            for i, y_node in enumerate(item['end']):
                heade = check_dup_node(same_var, y_node)
                if heade != None:
                    item['end'][i] = heade
    return ast_combine_edges


def build_hyper_edges(nodes, edges, index_map, index_map_ver, dup_nodes, same_vars):
    hyper_edges = []

    call_nodes = [node for node in nodes if node['type'] == 'CallExpression']
    for call_node in call_nodes:
        hyper_nodes = []
        hyper_nodes.append(call_node['key'])

        callee_edges = [e for e in edges if e['start'] == call_node['key'] and
                        get_nodes_by_key(nodes, e['end'])['type'] == 'Callee']
        if callee_edges:
            hyper_nodes.append(callee_edges[0]['end'])

        arg_list_edges = [e for e in edges if e['start'] == call_node['key'] and
                          get_nodes_by_key(nodes, e['end'])['type'] == 'ArgumentList']
        for arg_edge in arg_list_edges:
            args = [e['end'] for e in edges if e['start'] == arg_edge['end'] and
                    get_nodes_by_key(nodes, e['end'])['type'] == 'Argument']
            hyper_nodes.extend(args)

        hyper_nodes = list(set(hyper_nodes))
        if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_CALL']:
            hyper_edges.append({
                'type': 'HYPER_CALL',
                'nodes': hyper_nodes
            })

    loop_nodes = [node for node in nodes if node['type'] in ['ForStatement', 'WhileStatement', 'DoStatement']]
    for loop_node in loop_nodes:
        body_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                      e['start'] == loop_node['key'] and
                      get_nodes_by_key(nodes, e['end'])['type'] in ['CompoundStatement', 'ExpressionStatement']]

        if body_edges:
            hyper_nodes = [loop_node['key']]
            hyper_nodes.extend([e['end'] for e in body_edges])
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_LOOP']:
                hyper_edges.append({
                    'type': 'HYPER_LOOP',
                    'nodes': hyper_nodes
                })

    if_nodes = [node for node in nodes if node['type'] == 'IfStatement']
    for if_node in if_nodes:
        condition_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                           e['start'] == if_node['key'] and
                           get_nodes_by_key(nodes, e['end'])['type'] == 'Condition']

        branch_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                        e['start'] == if_node['key'] and
                        get_nodes_by_key(nodes, e['end'])['type'] in ['CompoundStatement', 'ExpressionStatement',
                                                                      'ElseStatement']]

        if condition_edges or branch_edges:
            hyper_nodes = [if_node['key']]
            hyper_nodes.extend([e['end'] for e in condition_edges])
            hyper_nodes.extend([e['end'] for e in branch_edges])
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_IF']:
                hyper_edges.append({
                    'type': 'HYPER_IF',
                    'nodes': hyper_nodes
                })

    switch_nodes = [node for node in nodes if node['type'] == 'SwitchStatement']
    for switch_node in switch_nodes:
        case_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                      e['start'] == switch_node['key'] and
                      get_nodes_by_key(nodes, e['end'])['type'] == 'Label']

        branch_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                        e['start'] == switch_node['key'] and
                        get_nodes_by_key(nodes, e['end'])['type'] in ['CompoundStatement']]

        if case_edges or branch_edges:
            hyper_nodes = [switch_node['key']]
            hyper_nodes.extend([e['end'] for e in case_edges])
            hyper_nodes.extend([e['end'] for e in branch_edges])
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_SWITCH']:
                hyper_edges.append({
                    'type': 'HYPER_SWITCH',
                    'nodes': hyper_nodes
                })

    return_nodes = [node for node in nodes if node['type'] == 'ReturnStatement']
    for return_node in return_nodes:
        return_expr_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                             e['start'] == return_node['key'] and
                             get_nodes_by_key(nodes, e['end'])['type'] in ['Expression', 'PrimaryExpression']]

        if return_expr_edges:
            hyper_nodes = [return_node['key']]
            hyper_nodes.extend([e['end'] for e in return_expr_edges])
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_RETURN']:
                hyper_edges.append({
                    'type': 'HYPER_RETURN',
                    'nodes': hyper_nodes
                })

    param_nodes = [node for node in nodes if node['type'] == 'Parameter']
    for param_node in param_nodes:
        decl_edges = [e for e in edges if e['type'] == 'IS_AST_PARENT' and
                      e['start'] == param_node['key'] and
                      get_nodes_by_key(nodes, e['end'])['type'] == 'IdentifierDecl']

        if decl_edges:
            hyper_nodes = [param_node['key']]
            hyper_nodes.extend([e['end'] for e in decl_edges])
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_PARAM']:
                hyper_edges.append({
                    'type': 'HYPER_PARAM',
                    'nodes': hyper_nodes
                })

    def_nodes = [node for node in nodes if node['type'] == 'IdentifierDecl']
    for def_node in def_nodes:
        def_key = def_node['key']
        uses = []

        for e in edges:
            if e['type'] == 'DEF' and e['end'] == def_key:
                uses.append(e['start'])

        for e in edges:
            if e['type'] == 'USE' and e['end'] == def_key:
                uses.append(e['start'])

        if len(uses) >= 2:  
            hyper_nodes = [def_key] + uses[:10] 
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_DATAFLOW']:
                hyper_edges.append({
                    'type': 'HYPER_DATAFLOW',
                    'nodes': hyper_nodes
                })

    control_nodes = [node for node in nodes if node['type'] in ['IfStatement', 'WhileStatement',
                                                                'ForStatement', 'SwitchStatement']]
    for ctrl_node in control_nodes:
        controlled = []
        for e in edges:
            if e['type'] == 'CONTROLS' and e['start'] == ctrl_node['key']:
                controlled.append(e['end'])

        if len(controlled) >= 2:
            hyper_nodes = [ctrl_node['key']] + controlled[:8]  # 限制节点数量
            hyper_nodes = list(set(hyper_nodes))

            if len(hyper_nodes) >= HYPER_EDGE_MIN_NODES['HYPER_CONTROL']:
                hyper_edges.append({
                    'type': 'HYPER_CONTROL',
                    'nodes': hyper_nodes
                })


    converted_hyper_edges = []
    for he in hyper_edges:
        try:
            node_indices = []
            for node_key in he['nodes']:
                if node_key not in index_map:
                    flag = False
                    for nodes in dup_nodes:
                        if node_key in nodes:
                            node_key = nodes[0]
                            flag = True
                            break
                    for same_var in same_vars:
                        if node_key in same_var:
                            node_key = same_var[0]
                            flag = True
                            break
                    if not flag:
                        max_key = len(index_map)
                        index_map[node_key] = max_key
                        index_map_ver[max_key] = node_key
                node_indices.append(index_map[node_key])

            if len(node_indices) >= HYPER_EDGE_MIN_NODES.get(he['type'], 2):
                converted_hyper_edges.append({
                    'type': he['type'],
                    'edge_type_id': edgeType_reduced[he['type']],
                    'nodes': node_indices
                })
        except Exception as ex:
            print(ex)
            print(he, 'key error')
            continue

    return converted_hyper_edges, index_map, index_map_ver

def build_hyper_normal_edges(nodes, edges, ):
    hyper_edges = []
    all_edges = list()
    call_nodes = [node for node in nodes if node['type'] == 'CallExpression'] 

    for node in call_nodes:
        caller_key = node['key']
        callee_edges = [e for e in edges if e['start'] == caller_key and
                        get_nodes_by_key(nodes, e['end'])['type'] == 'Callee']

        arg_list_edges = [e for e in edges if e['start'] == caller_key and
                          get_nodes_by_key(nodes, e['end'])['type'] == 'ArgumentList']

        all_args = []
        for arg_edge in arg_list_edges:
            arg_list_key = arg_edge['end']
            args = [e['end'] for e in edges if e['start'] == arg_list_key and
                    get_nodes_by_key(nodes, e['end'])['type'] == 'Argument']
            all_args.extend(args)

        if callee_edges or all_args:
            related_nodes = [caller_key]
            if callee_edges:
                related_nodes.append(callee_edges[0]['end']) 
            related_nodes.extend(all_args)  

            hyper_edges.append({
                'type': 'HYPER_CALL',
                'nodes': related_nodes
            })

    for node in nodes:
        if node['type'] in ['ForStatement', 'WhileStatement']:
            body_nodes = [e['end'] for e in edges if e['type'] == 'IS_AST_PARENT'
                          and e['start'] == node['key']
                          and get_nodes_by_key(nodes, e['end'])['type'] in ['CompoundStatement']]
            if body_nodes:
                hyper_edges.append({
                    'type': 'HYPER_LOOP',
                    'nodes': [node['key']] + body_nodes
                })
    if_nodes = [node for node in nodes if node['type'] == 'IfStatement']
    for node in if_nodes:
        condition_nodes = [e['end'] for e in edges
                           if e['start'] == node['key']
                           and get_nodes_by_key(nodes, e['end'])['type'] == 'Condition']

        branch_nodes = [e['end'] for e in edges
                        if e['start'] == node['key']
                        and get_nodes_by_key(nodes, e['end'])['type'] in ['CompoundStatement',
                                                                          'ExpressionStatement']]
        if condition_nodes or branch_nodes:
            hyper_edges.append({
                'type': 'HYPER_IF',
                'nodes': [node['key']] + condition_nodes + branch_nodes
            })

    switch_nodes = [node for node in nodes if node['type'] == 'SwitchStatement']
    for node in switch_nodes:
        case_nodes = [e['end'] for e in edges
                      if e['start'] == node['key']
                      and get_nodes_by_key(nodes, e['end'])['type'] == 'Label']

        branch_nodes = [e['end'] for e in edges
                        if e['start'] == node['key']
                        and get_nodes_by_key(nodes, e['end'])['type'] in ['CompoundStatement']]

        if case_nodes or branch_nodes:
            hyper_edges.append({
                'type': 'HYPER_SWITCH',
                'nodes': [node['key']] + case_nodes + branch_nodes
            })

    return_nodes = [node for node in nodes if node['type'] == 'ReturnStatement']
    for node in return_nodes:
        return_expr = [e['end'] for e in edges
                       if e['start'] == node['key']
                       and get_nodes_by_key(nodes, e['end'])['type'] in ['Expression', 'PrimaryExpression']]

        if return_expr:
            hyper_edges.append({
                'type': 'HYPER_RETURN',
                'nodes': [node['key']] + return_expr
            })

    param_nodes = [node for node in nodes if node['type'] == 'Parameter']
    for node in param_nodes:
        decl_nodes = [e['end'] for e in edges
                      if e['start'] == node['key']
                      and get_nodes_by_key(nodes, e['end'])['type'] == 'IdentifierDecl']

        if decl_nodes:
            hyper_edges.append({
                'type': 'HYPER_PARAM',
                'nodes': [node['key']] + decl_nodes
            })
    for he in hyper_edges:
        nodes_list = he['nodes']
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                all_edges.append([
                    nodes_list[i],
                    he['type'], 
                    nodes_list[j]
                ])
    return all_edges

def graphGeneration(nodes, edges, edge_type_map, ver_edge_type_map):
    index_map = dict()
    index_map_ver = dict()
    all_nodes = set()
    all_ast_edges = []

    all_edges = build_hyper_normal_edges(nodes, edges)
    for node in nodes:
        if node['isCFGNode'].strip() != 'True' or node['key'].strip() == 'File':
            continue
        all_nodes.add(node['key'])
        if node['type'] in ['CFGEntryNode', 'CFGExitNode']: 
            continue
        nodeKey = [node['key']]
        ast_edges = []
        build_ast(nodeKey, edges, ast_edges)

        if len(ast_edges) == 0:
            if spe_sent(node['type'], node['code'].strip().split()):
                dic = {}
                dic['start'] = nodeKey[0]
                dic['end'] = nodeKey
                ast_edges.append(dic)
            else:  
                return None, None, None, True, None
        all_ast_edges.append(ast_edges)

    edge_count_before = 0 
    node_count_before = set()  
    for item in all_ast_edges:
        for ast in item:
            node_count_before.add(ast['start'])  
            node_count_before = node_count_before | set(
                ast['end']) 
            edge_count_before += len(ast['end']) 

    dup_nodes = [] 
    step1_ast_edges = []  
    for i in all_ast_edges:  
        # break; continue; return
        if len(i) == 1 and i[0]['start'] == i[0]['end'][0] and len(i[0]['end']) == 1: 
            step1_ast_edges.append(i)  
        else:
            new_edges, dup_node =  ast_prune(nodes, i, 'IS_AST_PARENT')
            dup_nodes.extend(dup_node)
            step1_ast_edges.append(new_edges)

    same_var = get_same_var(step1_ast_edges, nodes)
    step2_ast_edges = var_combine(step1_ast_edges, same_var)

    edge_count_after = 0
    node_count_after = set()
    for item in step2_ast_edges:
        for ast in item:
            node_count_after.add(ast['start'])
            node_count_after = node_count_after | set(ast['end'])
            edge_count_after += len(ast['end'])

    nsc_edges_ = get_ncs_edges(step1_ast_edges, 'NSC', nodes)
    nsc_edges = get_combine_ncs_edges(step1_ast_edges, 'NSC', nodes, same_var)
    ast_type = 'IS_AST_PARENT'

    for item in step2_ast_edges:
        if len(item) == 1 and len(item[0]['end']) == 1 and item[0]['start'] == item[0]['end'][0]:
            continue
        for x in item:
            start = x['start']
            for end in x['end']:
                all_edges.append([start, ast_type, end])

    for e in edges:
        start, end, eType = e['start'], e['end'], e['type']
        start_node = get_nodes_by_key(nodes, start)
        end_node = get_nodes_by_key(nodes, end)
        if start_node['isCFGNode'].strip() != 'True' or end_node['isCFGNode'].strip() != 'True':
            continue
        if eType != 'IS_FILE_OF' and eType != ast_type:  
            if not eType in edge_type_map:
                continue
            all_edges.append([start, eType, end])

    for e in all_edges:
        start, _, end = e
        all_nodes.add(start)
        all_nodes.add(end)

    if len(all_nodes) == 0 or len(all_nodes) > 500:  
        return None, None, None, None, None

    for i, node in enumerate(all_nodes):  
        index_map[node] = i 
        index_map_ver[i] = node 

    all_edges_new = []  
    for e in all_edges:  
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)

    for e in nsc_edges:
        e_new = [index_map[e[0]], edge_type_map[e[1]], index_map[e[2]]]
        all_edges_new.append(e_new)

    loop = 'Self_loop'
    for node in all_nodes:
        self_loop = [index_map[node], edge_type_map[loop], index_map[node]]
        all_edges_new.append(self_loop)

    if len(all_edges_new) == 0:
        return None, None, None, None, None

    hyper_edges, index_map, index_map_ver = build_hyper_edges(nodes, edges, index_map, index_map_ver, dup_nodes, same_var)

    edges_num = {}
    for t in edge_type_map.keys():
        edges_num[t] = 0

    for e in all_edges_new:
        key = ver_edge_type_map[str(e[1])]
        edges_num[key] += 1

    edges_num['ast_reduced'] = edge_count_before - edge_count_after
    edges_num['nsc_reduced'] = len(nsc_edges_) - len(nsc_edges)
    edges_num['nodes_reduced'] = len(node_count_before) - len(node_count_after)

    hyper_edge_counts = {}
    for he in hyper_edges:
        edge_type = he['type']
        if edge_type not in hyper_edge_counts:
            hyper_edge_counts[edge_type] = 0
        hyper_edge_counts[edge_type] += 1

    edges_num['hyper_edges'] = hyper_edge_counts
    edges_num['total_hyper_edges'] = len(hyper_edges)

    return index_map_ver, all_edges_new, len(index_map_ver), edges_num, hyper_edges


def word2vec(nodes, index_map, graph, wv):
    gInput = list()
    all_nodes = set()
    for item in graph:
        s, _, e = item
        all_nodes.add(e)
        all_nodes.add(s)
    for i in index_map:
        true_id = index_map[i]
        node = get_nodes_by_key(nodes, true_id)
        node_content = node['code'].strip()
        if node_content.startswith('"') and node_content.endswith('"') and len(node_content) >= 2:
            node_content = node_content[1:-1]
        tokens = my_tokenizer(node_content)
        nrp = np.zeros(100)
        for token in tokens:
            try:
                embedding = wv.wv[token]
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)
        if len(tokens) > 0:
            fnrp = np.divide(nrp, len(tokens))
        else:
            fnrp = nrp
        gInput.append(fnrp.tolist())
    return gInput

if __name__ == '__main__':
    base_dir = '/data/AIinspur02/linshi001/dataset/'
    dataset_split = 'devign_data_split0407/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='normalized csv files to process',
                        default=base_dir + 'devign_joern')  
    parser.add_argument('--src', help='source c files to process',
                        default=base_dir + 'devign_raw_code') 
    parser.add_argument('--json_files', help='train and test and valid',
                        default=[base_dir + dataset_split + 'train.json',
                                 base_dir + dataset_split + 'test.json',
                                 base_dir + dataset_split + 'valid.json'])
    parser.add_argument('--wv',
                        default='/data/AIinspur02/linshi001/dataset/devign_wv_models/devign_train_subtoken_data_myself')  # word2vec
    parser.add_argument('--output_dir',
                        default=base_dir + 'hyper_w2v_7edges_RealHyper0413')
    args = parser.parse_args()
    model = Word2Vec.load(args.wv)
    train_path, test_path, valid_path = args.json_files 
    train_data = []
    test_data = []
    valid_data = []
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    with open(valid_path, 'r') as f:
        valid_data = json.load(f)
    data = [train_data, test_data, valid_data]  
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("*" * 100)

    train_output_path = open(os.path.join(output_dir, 'devign-train-v2.json'), 'w')
    test_output_path = open(os.path.join(output_dir, 'devign-test-v2.json'), 'w')
    valid_output_path = open(os.path.join(output_dir, 'devign-valid-v2.json'), 'w')
    output_files = [train_output_path, test_output_path, valid_output_path]

    train_num_path = open(os.path.join(output_dir, 'devign-train-num-v2.json'), 'w')
    test_num_path = open(os.path.join(output_dir, 'devign-test-num-v2.json'), 'w')
    valid_num_path = open(os.path.join(output_dir, 'devign-valid-num-v2.json'), 'w')
    num_files = [train_num_path, test_num_path, valid_num_path]

    train_file = open(os.path.join(output_dir, 'devign-train-file.json'), 'w')
    test_file = open(os.path.join(output_dir, 'devign-test-file.json'), 'w')
    valid_file = open(os.path.join(output_dir, 'devign-valid-file.json'), 'w')
    file_names = [train_file, test_file, valid_file]

    bad_file = []
    bad_file_path = open(os.path.join(output_dir, 'bad_file.json'), 'w')
    # train、test、valid json file
    for i in range(len(data)):
        print("!!!")
        final_data = []
        final_num = []
        files = []
        num = 0
        for _, entry in enumerate(tqdm(data[i])):
            file_name = str(entry['id']) + '.c'  
            nodes_path = os.path.join(args.csv, file_name, 'tmp', file_name, 'nodes.csv')  
            edges_path = os.path.join(args.csv, file_name, 'tmp', file_name, 'edges.csv')
            label = int(entry['target'])  
            if not os.path.exists(nodes_path) or not os.path.exists(edges_path):  
                continue
            nodes = read_csv(nodes_path)  
            edges = read_csv(edges_path)
            index_map, graph, nodes_num, edges_num, hyper_edges = graphGeneration(nodes, edges, edgeType_reduced,
                                                                                  ver_edge_type_reduced)
            if index_map is None or graph is None or nodes_num is None or edges_num is None:
                continue

            if hyper_edges is None:
                hyper_edges = []

            gInput = word2vec(nodes, index_map, graph, model)
            if gInput is None:
                continue

            if check(index_map, graph, gInput) != True:
                print("check error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            data_point = {
                'node_features': gInput,
                'graph': graph,  
                'hyper_edges': hyper_edges,  
                'targets': [[label]],
                'origin_id': entry['id']
            }

            num_point = {
                'file_name': file_name,
                'nodes_num': nodes_num,
                'edges_num': edges_num
            }
            num += 1
            files.append(file_name)
            final_data.append(data_point)
            final_num.append(num_point)

        json.dump(final_data, output_files[i])
        json.dump(final_num, num_files[i])
        json.dump(files, file_names[i])
        output_files[i].close()
        num_files[i].close()
        file_names[i].close()

    json.dump(bad_file, bad_file_path)
    bad_file_path.close()
