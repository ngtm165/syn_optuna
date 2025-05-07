from rdkit import Chem
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
import torch
import sys
import os
from pathlib import Path
import pickle
import gzip
import random
import networkx as nx 

root_dir = str(Path(__file__).resolve().parents[1])
os.chdir(root_dir)

atom_list = list(range(1, 119))
charge_list = [-2, -1, 0, 1, 2, 'other']
hybridization = ['SP', 'SP2', 'SP3', 'other']
valence = [1, 2, 3, 4, 5, 6, 'other']

bond_type1 = [1, 2, 3, 'other']
bond_type2 = ['normal', 'increase', 'decrease', 'other']

def read_data(data_path, graph_path, target):
    graph, labels = [], []
    data = pd.read_csv(data_path)
    labels_lst = data[target].tolist()  # Directly get the target column as a list
    with gzip.open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    # graphs_lst = [i['ITSGraph'][2] for i in graphs]
    graphs_lst = list(graphs.values())  # Chuyển đổi values() thành list
    
    return graphs_lst, labels_lst

def one_hot(idx, length):
    lst_onehot = [0 for i in range(length)]
    lst_onehot[idx] = 1
    return lst_onehot

def neighbors_to_quantum_numbers(neighbors):
    element_to_atomic_number_1 = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Br': 35
    }
    if not neighbors:  # Kiểm tra nếu danh sách rỗng
        return [0]
    neighbor_atomic_numbers = []
    for element in neighbors:
        if element not in element_to_atomic_number_1:
            neighbor_atomic_numbers.append(0)  # Thêm 0 nếu không tìm thấy
        else:
            neighbor_atomic_numbers.append(element_to_atomic_number_1[element])
    return neighbor_atomic_numbers

def element_to_quantum_numbers(element):
    element_to_atomic_number = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Br': 35
    }
    if element not in element_to_atomic_number:
        return None
    atomic_number = element_to_atomic_number[element]
    return atomic_number_to_quantum_numbers(atomic_number)

def atomic_number_to_quantum_numbers(atomic_number):
    electron_configuration = get_electron_configuration(atomic_number)
    outer_subshell = electron_configuration[-1]
    # print(atomic_number)
    n = int(outer_subshell[0])
    l = 0 if outer_subshell[1] == 's' else 1 if outer_subshell[1] == 'p' else 2 if outer_subshell[1] == 'd' else 3

    num_orbitals = 2 * l + 1
    num_electrons = int(outer_subshell[2:]) # Cần tính số electron trong phân lớp ngoài cùng

    orbitals = [0] * num_orbitals
    spin = 1
    # last_orbital_index = 0  # Theo dõi vị trí electron cuối cùng
    last_spin = 1 # Thêm biến để theo dõi spin electron cuối cùng


    for i in range(num_electrons):
        orbital_index = i % num_orbitals
        if orbitals[orbital_index] == 0:
            orbitals[orbital_index] = spin
            last_spin = spin # Cập nhật spin electron cuối cùng
        else:
            orbitals[orbital_index] = 2
            last_spin = -spin # Cập nhật spin electron cuối cùng
        # spin *= -1

    # Ánh xạ orbital_index sang ml
    ml_map = list(range(-l, l + 1))  # Tạo danh sách [-l, -l+1, ..., l-1, l]
    ml = ml_map[orbital_index]

    # Xác định ms dựa trên spin cuối cùng
    ms = 0.5 if last_spin == 1 else -0.5
    
    empty_orbitals = []
    single_electron_orbitals = []
    full_orbitals = []

    for i, orbital_state in enumerate(orbitals):
        if orbital_state == 0:
            empty_orbitals.append(ml_map[i])
        elif orbital_state == 1 or orbital_state == -1 :
            single_electron_orbitals.append(ml_map[i])
            

    for i, orbital_state in enumerate(orbitals):
        if orbital_state == 2:
            full_orbitals.append(ml_map[i])

    
    # Tính tổng số electron lớp ngoài cùng
    outer_electrons = 0
    for subshell in electron_configuration:
        if int(subshell[0]) == n:  # Kiểm tra nếu phân lớp thuộc lớp ngoài cùng
            outer_electrons += int(subshell[-1])
    
    # Tính tổng số orbital lớp ngoài cùng
    outer_orbitals = 0

    max_l = []
    if n >= 1:
        for i in range(min(n, 4)):  # Chỉ lấy tối đa 4 giá trị của l
            max_l.append(i)
            outer_orbitals += 2 * i + 1
            
    # xac dinh hoa tri
    e = outer_orbitals if outer_electrons > outer_orbitals else outer_electrons
        #Thêm logic xử lý riêng cho atomic_number 8 và 9 nếu cần.
    if atomic_number == 8 or atomic_number == 9:
        e = len(single_electron_orbitals)
    
    return (n, l, ml, ms, e)


def get_electron_configuration(atomic_number):
    subshells = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p']
    max_electrons = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6]
    electron_configuration = []
    remaining_electrons = atomic_number
    for i in range(len(subshells)):
        if remaining_electrons <= 0:
            break
        if remaining_electrons <= max_electrons[i]:
            electron_configuration.append(subshells[i] + str(remaining_electrons))
            remaining_electrons = 0
        else:
            electron_configuration.append(subshells[i] + str(max_electrons[i]))
            remaining_electrons -= max_electrons[i]
    return electron_configuration

def hybridization_to_spdf(hybridization):
    hybridization = hybridization.lower()

    s = hybridization.count('s')
    p = hybridization.count('p')
    d = hybridization.count('d')
    f = hybridization.count('f')

    p_num = 0
    d_num = 0
    f_num = 0

    if 'p' in hybridization:
        p_index = hybridization.find('p')
        if p_index + 1 < len(hybridization) and hybridization[p_index + 1].isdigit():
            num_str = ''
            for char in hybridization[p_index + 1:]:
                if char.isdigit():
                    num_str += char
                else:
                    break
            if num_str:
                p_num = int(num_str)
            else:
                p_num = 1

    if 'd' in hybridization:
        d_index = hybridization.find('d')
        if d_index + 1 < len(hybridization) and hybridization[d_index + 1].isdigit():
            num_str = ''
            for char in hybridization[d_index + 1:]:
                if char.isdigit():
                    num_str += char
                else:
                    break
            if num_str:
                d_num = int(num_str)
            else:
                d_num = 1

    if 'f' in hybridization:
        f_index = hybridization.find('f')
        if f_index + 1 < len(hybridization) and hybridization[f_index + 1].isdigit():
            num_str = ''
            for char in hybridization[f_index + 1:]:
                if char.isdigit():
                    num_str += char
                else:
                    break
            if num_str:
                f_num = int(num_str)
            else:
                f_num = 1

    total = s + p_num + d_num + f_num
    
    return [s, p_num, d_num, f_num], total
    
    # if total == 0:
    #   return [0,0,0,0]

    # return [s / total, p_num / total, d_num / total, f_num / total], total

def lone_pairs (total, sigma):
    lone = total - sigma 
    return lone 

def add_vectors(a, b):

    if len(a) != len(b):
        raise ValueError("Hai vectơ phải có cùng chiều dài.")

    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result

def calculate_standard_order(graph, standard_order):
    """Tính tổng standard order từ thông tin đồ thị."""
    standard_orders = []
    for u, v, data in graph.edges(data=True):
        standard_orders.append(data['standard_order'])
    return sum(standard_orders)


class ReactionDataset(Dataset):
    def __init__(self, data_path, graph_path, target):
        super(Dataset, self).__init__()
        self.graph, self.labels = read_data(data_path, graph_path, target)

    def __getitem__(self, index):
        graph = self.graph[index]
        lst_nodes = list(graph.nodes())
        lst_nodes_update = [lst_nodes.index(i) for i in lst_nodes]
        lst_edges = list(graph.edges())
        lst_edges_update = [(lst_nodes.index(u), lst_nodes.index(v)) for u, v in lst_edges]
        label = self.labels[index]

        #atom features
        pt = Chem.GetPeriodicTable()
        # atom_fea_graph = []
        max_neighbors = 6  # Tìm số neighbors lớn nhất. viết kèm 3 dòng dưới
        
        atom_fea_graph = []
        atom_hybrid_change = []
        hcount_change = []
        hybrid_change = []

        for i in lst_nodes:
            
            atom_data = graph.nodes(data=True)[i] #ver 7_mới

            #charge
            charge_1 = atom_data['typesGH'][0][3] 
            charge_2 = atom_data['typesGH'][1][3]
            charge_atom = [charge_1] + [charge_2] #ver_3
            charge_change = charge_1 - charge_2
        

            #hybridization
            hybrid_1 = atom_data['typesGH'][0][4] 
            hybrid_2 = atom_data['typesGH'][1][4] 

            atom_hybrid_1, total_1 = hybridization_to_spdf(hybrid_1)
            atom_hybrid_2, total_2 = hybridization_to_spdf(hybrid_2)

            atom_hybrid = atom_hybrid_1 + atom_hybrid_2
            hybrid_change = add_vectors (atom_hybrid_1, atom_hybrid_2)

            # Tích hợp số lượng tử
            element = graph.nodes(data=True)[i]['element']
            quantum_numbers = element_to_quantum_numbers(element)
            if quantum_numbers:
                n, l, ml, ms, e = quantum_numbers
                quantum_features = [n, l, ml, ms]  # Chuyển thành list
                n_onehot = one_hot(n - 1, 7)  # Giả sử n tối đa là 7
                l_onehot = one_hot(l, 4)  # l có thể từ 0 đến 3
                ml_onehot = one_hot(ml + 3, 7)  # Giả sử ml có thể từ -3 đến 3
                ms_onehot = [1, 0] if ms == 0.5 else [0, 1]
            else:
                quantum_features = [0, 0, 0, 0]  # Giá trị mặc định nếu không tìm thấy
                n_onehot = [0] * 7
                l_onehot = [0] * 4
                ml_onehot = [0] * 7
                ms_onehot = [0, 0]
                
            quantum_onehots = n_onehot + l_onehot + ml_onehot + ms_onehot

            # Liên kết tối đa (valence electrons)
            e_max = [e]
            
            # Mã hóa one-hot cho các thuộc tính bổ sung
            hcount_1 = atom_data['typesGH'][0][2]
            hcount_2 = atom_data['typesGH'][1][2]
        
            # # Featurize số lượng nguyên tố neighbors

            neighbor_1 = atom_data['typesGH'][0][5] 
            neighbor_2 = atom_data['typesGH'][1][5] 

            neighbor_count_1 = len(neighbor_1)
            neighbor_count_2 = len(neighbor_2)
            neighbor_change = neighbor_count_1 - neighbor_count_2

            neighbor_elements_1 = neighbors_to_quantum_numbers(neighbor_1)
            neighbor_elements_2 = neighbors_to_quantum_numbers(neighbor_2)

            #Kiểm tra thuyết
            h_val_1 = neighbor_elements_1.count(1)
            h_val_2 = neighbor_elements_2.count(1)
            h_1 = [hcount_1 + h_val_1] 
            h_2 = [hcount_2 + h_val_2] 
            hcount = [hcount_1 + h_val_1] + [hcount_2 + h_val_2] 

            hcount_change = add_vectors (h_1, h_2)

            if h_val_1 == 0:
                sigma_1 = hcount_1 + h_val_1 + neighbor_count_1
            else: sigma_1 = neighbor_count_1

            if h_val_2 == 0:
                sigma_2 = hcount_2 + h_val_2 + neighbor_count_2   
            else: sigma_2 = neighbor_count_2         

            lone_1 = lone_pairs (total_1, sigma_1)
            lone_2 = lone_pairs (total_2, sigma_2)

            atom_hybrid_p_1 = atom_hybrid_1 + [sigma_1] + [lone_1]
            atom_hybrid_p_2 = atom_hybrid_2 + [sigma_2] + [lone_2]
            atom_hybrid_p = atom_hybrid_p_1 + atom_hybrid_p_2
            atom_hybrid_change = add_vectors (atom_hybrid_p_1, atom_hybrid_p_2)
            
            atom_fea = quantum_features + e_max + [charge_1] + [charge_change] + h_1 + hcount_change + atom_hybrid_p_1 + atom_hybrid_change + [neighbor_count_1] + [neighbor_change]  
            atom_fea_graph.append(atom_fea)

        
        #bond_feature
        row, col, edge_feat_graph=[], [], []
        for idx, bond in enumerate(lst_edges_update): # bond là cặp (chỉ số nút u, chỉ số nút v)
            u = bond[0] # Chỉ số nút nguồn cho hướng u -> v
            v = bond[1] # Chỉ số nút nguồn cho hướng v -> u

            row += [u, v]
            col += [v, u]

            # # Thêm các đặc trưng cạnh mới
            order_0, order_1 = list(graph.edges(data=True))[idx][2]['order']
            standard_order = list(graph.edges(data=True))[idx][2]['standard_order']
            
            changes = []

            # Kiểm tra thuộc liên hợp
            con_0, con_1 = list(graph.edges(data=True))[idx][2]['conjugated']
            bond_con_0 = [1] if con_0 == True else [0]
            bond_con_1 = [1] if con_1 == True else [0]

            if order_0 == 1:
                edge_fea1 = [1,0] + bond_con_0
            elif order_0 == 2:
                edge_fea1 = [1,1] + bond_con_0
            elif order_0 == 3:
                edge_fea1 = [1,2] + bond_con_0
            elif order_0 == 1.5:   
                edge_fea1 = [1,0.5] + bond_con_0
            else:
                edge_fea1 = [0,0,0]
            
            if order_1 == 1:
                edge_fea2 = [1,0] + bond_con_1
            elif order_1 == 2:
                edge_fea2 = [1,1] + bond_con_1
            elif order_1 == 3:
                edge_fea2 = [1,2] + bond_con_1
            elif order_1 == 1.5:  
                edge_fea2 = [1,0.5] + bond_con_1
            else:
                edge_fea2 = [0,0,0]
            
            changes = add_vectors (edge_fea1, edge_fea2) #signma changes, pi changes, conjugated changes
            # print (changes)

            if standard_order == 0 and order_0 == order_1: #unchaged
                edge_fea3 = edge_fea1 + changes[:2]
            elif standard_order > 0 or standard_order < 0: 
                edge_fea3 = edge_fea1 + changes[:2] if order_0 > order_1 else edge_fea2 + changes[:2]
            else: edge_fea3 = [0,0,0,0,0]

            total_standard_order = calculate_standard_order(graph, standard_order)

            # Tính toán edge_fea5 dựa trên tổng standard order
            if total_standard_order == 0:
                edge_fea5 = [0]
            elif total_standard_order == 1:
                edge_fea5 = [1]
            elif total_standard_order == -1:
                edge_fea5 = [-1]
            else:
                edge_fea5 = [total_standard_order] # handle other cases

            #Kiểm tra thuộc aromatic
            aromatic_val = graph.nodes(data=True)[list(graph.edges(data=True))[idx][0]].get('aromatic', True)
            aromatic_onehot = [1] if aromatic_val else [0]
            if aromatic_onehot == [1] and order_0 == 1.5 and order_1 == 1.5:
                edge_aromatic = [1]
            else:
                edge_aromatic = [0]
            
            # print(edge_fea3)
            edge_fea = edge_fea1 + edge_fea2 #+ edge_aromatic #edge_fea1 + edge_fea2 + edge_fea3 ##edge_fea1 + edge_fea2 + changes[:2]

            # --- THAY ĐỔI CHÍNH Ở ĐÂY ---

            # Lấy đặc trưng của hai nguyên tử tham gia liên kết
            atom_feat_u = atom_fea_graph[u] 
            atom_feat_v = atom_fea_graph[v]

            # Tạo đặc trưng có hướng cho u -> v: Ghép đặc trưng nguyên tử nguồn u với đặc trưng liên kết
            directed_feature_uv = np.concatenate((atom_feat_u, edge_fea)).tolist() # Ví dụ dùng numpy concatenate
            edge_feat_graph.append(directed_feature_uv)

            # Tạo đặc trưng có hướng cho v -> u: Ghép đặc trưng nguyên tử nguồn v với đặc trưng liên kết
            directed_feature_vu = np.concatenate((atom_feat_v, edge_fea)).tolist() # Ví dụ dùng numpy concatenate
            edge_feat_graph.append(directed_feature_vu)


        edge_index=torch.tensor([row,col])
        edge_attr=torch.tensor(np.array(edge_feat_graph),dtype=torch.float)
        node_attr=torch.tensor(np.array(atom_fea_graph),dtype=torch.float)
        y=torch.tensor(label,dtype=torch.float)
        data= Data(x=node_attr,y=y,edge_index=edge_index,edge_attr=edge_attr) ##thử bỏ ',edge_attr=edge_attr'

        return data

    def __len__(self):
        return len(self.graph)

def main():
    
    data_path='./Data/regression/phosphatase/phosphatase.csv'
    graph_path='./Data/regression/phosphatase/its_new/phosphatase.pkl.gz'
    target='Conversion'
    graphdata=ReactionDataset(data_path,graph_path,target)
    print(graphdata.__getitem__(8))

if __name__=='__main__':
    main()
