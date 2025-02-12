from collections import defaultdict
import json
import jsonlines
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import math
from operator import itemgetter

def extract_user(datapath,setclass):
    with open(f'./{datapath}/entity2id.json', "r+", encoding='utf-8') as f1:
        entity2id = json.load(f1)
    with open(f"./{datapath}/{setclass}_data_dbpedia.jsonl", "r+", encoding="utf8") as f:
        u_i_dict=defaultdict()
        m = set()
        user_set=set()
        for line in tqdm(jsonlines.Reader(f),desc="extracting user-movies"):
            # 提取用户id
            user = line["initiatorWorkerId"]

            # 提取用户所相关的所有电影
            new_movie, new_movie_name = [], []
            
            for i,message in enumerate(line["messages"]):
                for j, movie in enumerate(message['movie']):
                    if movie in entity2id:
                        movie_ids = [entity2id[movie]]
                        new_movie.extend(movie_ids)
                        new_movie_name.extend([message['movie_name'][j]])
                        m.add(message['movie_name'][j])
                        # 接下来转化为movie_id
            user_set.add(user)
    return user_set

def get_movie_set(datapath,setclass):
    
    fo = open(f"./{datapath}/{setclass}.txt","a+")
    fo.write("conversationId"+"\t"+"userId"+"\t"+"items"+"\n")
    
    with open(f'./{datapath}/entity2id.json', "r+", encoding='utf-8') as f1:
        entity2id = json.load(f1)
    movie_set=set()
    with open(f"./{datapath}/{setclass}_data_dbpedia.jsonl", "r+", encoding="utf8") as f:
        for line in tqdm(jsonlines.Reader(f),desc="extracting user-movies"):
            # 提取用户id
            user = line["initiatorWorkerId"]
            # 提取对话id
            conv_id = line["conversationId"]
            # fo.write(str(conv_id)+"\t"+str(user)+"\t")
            # 提取用户所相关的所有电影
            new_movie= []
            for i,message in enumerate(line["messages"]):
                for j, movie in enumerate(message['movie']):
                    if movie in entity2id:
                        movie_ids = entity2id[movie]
                        # print(movie_ids)
                        movie_set.add(movie_ids)

                        # fo.write(str(movie_ids)+",")
            # 然后用正则表达式“,$”，把末尾的逗号去掉
    with open(f'{setclass}_movie_set.pkl', 'wb') as file:
        pickle.dump(movie_set, file)
    return movie_set    

def sort_dict_by_value(dictionary):
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    with open("sorted_item_users_dict.json", "w") as file:
        json.dump(sorted_dict, file)
    return sorted_dict

def itemCF_by_user_nums(datapath,k):
    """
    calculate user similarity
    datapath:所在路径
    setclass：训练集、测试集、验证集
    
    output：train_u2u_sim.pkl
    """
    vid_user = {}
    user_sim_matrix = {}
    uid_vcount = {}
    with open(f'./{datapath}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)
        # print(session_data[6])
    for uid in tqdm(session_data):
        # uid用户的所有历史会话
        u_sess = session_data[uid]
        # 如果uid不存在于字典中，则value默认为一个set()
        uid_vcount.setdefault(uid, set())
        # 取历史中的一段会话
        for sess in u_sess:
            # 循环会话中的每个项目
            for vid in sess: # vid是项目
                # 建立项目-用户集的关系
                if vid not in vid_user:
                    vid_user[vid] = []
                vid_user[vid].append(uid)
    sort_dict_by_value(vid_user)

def itemCorrelate(dataset,k):
    uid_item = {}
    item_sim_matrix = {}
    vid_ucount = {}
    with open(f'./{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        uid_item[uid] = set()
        # uid_vcount.setdefault(uid,set())
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
            result[item] = [item[0] for item in topK]
    # print(result)
    return result

def extract_last_n_items(dataset_name,num,sample_size):
    adj2 = [dict() for _ in range(num)]
    adj_in = [[] for _ in range(num)]
    adj_out = [[] for _ in range(num)]
    # A0-A-B
    relation_out = [] 
    relation_in = [] 

    with open(f'./redial/train.pkl', 'rb') as f:
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

    for source, targets in adj1.items():
        max_count = max(targets.values())
        max_value = [value for value, count in targets.items() if count == max_count]
        adj1[source] = max_value[0]
        if(source == max_value[0] and len(max_value)>1):
            adj1[source] = max_value[1]

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
        if(source == max_value[0] and len(max_value)>1):
            adj2[source] = max_value[1]
    # print(adj2)
    return adj1,adj2

def subkg(item_topK,adj1,adj2):
    res = defaultdict(list)

    for item, succeed_item in adj1.items():
        res[str(item)].append([0, succeed_item])

    for item, pre_item in adj2.items():
        res[str(item)].append([1, pre_item])

    for item, related_item in item_topK.items():
        for single_relateditem in related_item:
            res[str(item)].append([2, single_relateditem])
    res = {key.replace("'", '"'): value for key, value in res.items()}

    # 存入jsonl
    with open('entity_kg_items.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)

if __name__ == "__main__":
    redial_path = "redial"
    inspired_path = "inspired"
    k = 5 # 可调整超参数
    set_li = ["train","valid","test"]
    # extract_movie(redial_datapath,setclass)
    # userCF(redial_datapath,setclass,k)
    # user_user_edge(redial_datapath,setclass,k)
    # train_set = get_movie_set(redial_datapath,"train")
    # valid_set = get_movie_set(redial_datapath,"valid")
    # test_set = get_movie_set(redial_datapath,"test")
    # # 取电影交集
    # # set_train_test = set(train_set) & set(valid_set)
    # set_testNot_InTrain=(test_set-train_set) # 410
    # set_validNot_InTrain=(valid_set-train_set) # 254
    # set_testNot_InValid=(set_testNot_InTrain-valid_set) # 87

    # # print(set_validNot_InTrain)
    # print(len(train_set)) # 5638
    # print(len(valid_set)) # 1940
    # print(len(test_set)) # 1927
    # print(len(set_testNot_InValid)) #389

    # train_user_set = extract_user(redial_datapath,"train")
    # valid_user_set = extract_user(redial_datapath,"valid")
    # test_user_set = extract_user(redial_datapath,"test")
    # print(len(train_user_set & valid_user_set))
    # print(len(train_user_set & test_user_set)) 
    # print(len(test_user_set & valid_user_set))
    # itemCF_by_user_nums(redial_datapath,1)
    # itemCF_by_HGNN(redial_datapath,1)

    item_topK=itemCorrelate(redial_path,k)
    # 提取训练集中，每个项目的top-1个后继项目结点adj1,每个项目的top-1个前驱节点adj2
    adj1,adj2 = extract_last_n_items(redial_path,2,1)
    # 根据item_topK和adj1、adj2构图
    subkg(item_topK,adj1,adj2)
