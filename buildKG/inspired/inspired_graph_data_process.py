from collections import defaultdict
import json
import jsonlines
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import math
from operator import itemgetter
from collections import defaultdict

dialo_id_dict=dict()
def extract_movie(datapath,setclass):
    with open(f'./{datapath}/entity2id.json', "r+", encoding='utf-8') as f1:
        entity2id = json.load(f1)
    with open(f"./{datapath}/{setclass}_dbpedia_more_info.jsonl", "r+", encoding="utf8") as f:
        u_i_dict=dict()
        for line in tqdm(jsonlines.Reader(f),desc="extracting inspired user-movies"):
            session = line[0]
            user = session["seeker_id"]
            movie_ids=[] 
            # 对每段对话进行遍历
            for sess in line:
                curr_movie_list=sess["movie_link"] # 当前的电影link列表
                for ml in curr_movie_list:
                    movie_id = entity2id[ml]
                    movie_ids.extend([movie_id])
            if(u_i_dict.__contains__(user)):
                u_i_dict[user].append(movie_ids)
            else:
                u_i_dict[user]=[movie_ids]

        with open(f"./{datapath}/{setclass}.pkl", 'wb') as f2:
            pickle.dump(u_i_dict,f2,-1)

def get_movie_set(datapath,setclass):
    
    with open(f'./{datapath}/entity2id.json', "r+", encoding='utf-8') as f1:
        entity2id = json.load(f1)
    movie_set=set()
    with open(f"./{datapath}/{setclass}_dbpedia_more_info.jsonl", "r+", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f),desc="extracting user-movies"):
            user = line[0]["seeker_id"]
            user = user.lstrip('user_') 
            
            conv_id = line[0]["dialog_id"]
            conv_id = dialo_id_dict[conv_id]
            for sess in line:
                curr_movie_list=sess["movie_link"] # 当前的电影link列表
                for ml in curr_movie_list:
                    movie_id = entity2id[ml]
                    movie_set.add(movie_id)

    with open(f'{setclass}_movie_set.pkl', 'wb') as file:
        pickle.dump(movie_set, file)
    return movie_set         

def dialogToID():
    id_value = 0
    global dialo_id_dict
    with open(f"./data/train_dbpedia_more_info.jsonl", "r+", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f),desc="extracting user-movies"):
            # 提取对话id
            conv_id = line[0]["dialog_id"]
            if(dialo_id_dict.__contains__(conv_id)):
                continue
            else:
                id_value+=1
                dialo_id_dict[conv_id]=id_value

    with open(f"./data/valid_dbpedia_more_info.jsonl", "r+", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f),desc="extracting user-movies"):
            # 提取对话id
            conv_id = line[0]["dialog_id"]
            if(dialo_id_dict.__contains__(conv_id)):
                continue
            else:
                id_value+=1
                dialo_id_dict[conv_id]=id_value

    with open(f"./data/test_dbpedia_more_info.jsonl", "r+", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f),desc="extracting user-movies"):
            # 提取对话id
            conv_id = line[0]["dialog_id"]
            if(dialo_id_dict.__contains__(conv_id)):
                continue
            else:
                id_value+=1
                dialo_id_dict[conv_id]=id_value

def itemCorrelate(dataset,k):
    uid_item = {}
    item_sim_matrix = {}
    vid_ucount = {}
    with open(f'./{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        uid_item[uid] = set()
        for sess in u_sess:
            for vid in sess:
                uid_item[uid].add(vid)
                vid_ucount.setdefault(vid, set())
                vid_ucount[vid].add(uid)

    for uid, items in tqdm(uid_item.items()):
        for v in items:
            for _v in items:
                if _v == v:
                    continue
                item_sim_matrix.setdefault(v, {})
                item_sim_matrix[v].setdefault(_v, 0)
                item_sim_matrix[v][_v] += (1 / len(items))
    for v, related_items in item_sim_matrix.items():
        for _v, count in related_items.items():
            item_sim_matrix[v][_v] = count / math.sqrt(len(vid_ucount[v]) * len(vid_ucount[_v]))
    item_topK = {}
    
    for item in item_sim_matrix:
        item_topK[item] = sorted(item_sim_matrix[item].items(), key=itemgetter(1), reverse=True)[0:k]
    
    result = {}
    for item, topK in item_topK.items():
        if topK:
            result[item] = topK[0][0]
    return result

def extract_last_n_items(dataset_name,num,sample_size):
    # adj1 = [dict() for _ in range(num)]
    adj2 = [dict() for _ in range(num)]
    adj_in = [[] for _ in range(num)]
    adj_out = [[] for _ in range(num)]
    # A0-A-B
    relation_out = [] 
    relation_in = [] 

    with open(f'./data/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    for u in tqdm(graph, desc='build the graph...', leave=False):
        u_seqs = graph[u]
        for s in u_seqs:
            for i in range(len(s)-1):
                relation_out.append([s[i], s[i + 1]])
                relation_in.append([s[i + 1], s[i]])
    # print(relation_in)
    adj1 = defaultdict(dict)
    for tup in relation_out:
        source = tup[0]
        target = tup[1]
        if target in adj1[source]:
            adj1[source][target] += 1
        else:
            adj1[source][target] = 1
    # print(adj1)

    for source, targets in adj1.items():
        max_count = max(targets.values())
        max_value = [value for value, count in targets.items() if count == max_count]
        adj1[source] = max_value[0]

    adj2 = defaultdict(dict)

    for tup in relation_in:
        source = tup[0]
        target = tup[1]
        if target in adj2[source]:
            adj2[source][target] += 1
        else:
            adj2[source][target] = 1

    for source, targets in adj2.items():
        max_count = max(targets.values())
        max_value = [value for value, count in targets.items() if count == max_count]
        adj2[source] = max_value[0]
    # print(adj2)
    return adj1,adj2

def subkg(item_topK,adj1,adj2):
    
    res = defaultdict(list)

    for item, succeed_item in adj1.items():
        res[str(item)].append([0, succeed_item])

    for item, pre_item in adj2.items():
        res[str(item)].append([1, pre_item])

    for item, related_item in item_topK.items():
        res[str(item)].append([2, related_item])
    res = {key.replace("'", '"'): value for key, value in res.items()}

    # 存入jsonl
    with open('entity_kg.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)
    # print(res)


if __name__ == "__main__":
    # redial_datapath = "redial"
    inspired_path = "data"
    # setclass = "train"
    # setclass = "valid"
    # setclass = "test"
    k = 5
    # set_li = ["train","valid","test"]
    # dialogToID()

    # train_set = get_movie_set(inspired_path,"train")
    # valid_set = get_movie_set(inspired_path,"valid")
    # test_set = get_movie_set(inspired_path,"test")
    # # 取电影交集
    # # set_train_test = set(train_set) & set(valid_set)
    # set_testNot_InTrain=(test_set-train_set) # 93
    # set_validNot_InTrain=(valid_set-train_set) # 112
    # set_testNot_InValid=(set_testNot_InTrain-valid_set) # 87
    # # print(set_testNot_InTrain)
    # print(len(train_set)) # 1273
    # print(len(valid_set)) # 283
    # print(len(test_set)) # 250
    # print(len(set_testNot_InValid)) 

    item_topK=itemCorrelate(inspired_path,5)
    adj1,adj2 = extract_last_n_items(inspired_path,2,1)
    # 根据item_topK和adj1、adj2构图
    subkg(item_topK,adj1,adj2)





