import numpy as np
import os
def getnpy(path):
    f=open(path,"r")
    cards=f.readlines()
    ans=[]
    for card in cards:
        ans.append(float(card))
    return np.array(ans,dtype=np.float64)

def get_important_idx(mod,work_load_path,true_card_path):
    msk=[]
    iot_idx=mskmp.get(mod)
    f=open(true_card_path,"r")
    cards=f.readlines()
    for idx,card in enumerate(cards):
        if float(card)>1e8:
            iot_idx.append(idx+1)
    # if mod=='ins_heavy':
    #     iot_idx=[117]
    # print(iot_idx)
    nowquery=0
    nowidx=0
    f=open(work_load_path,"r")
    querys=f.readlines()
    for query in querys:
        if query.startswith("test_query"):
            nowquery+=1
        elif query.startswith("test_sub_query"):
            if nowquery in iot_idx:
                msk.append(nowidx)
            nowidx+=1
    return msk

# methods=["baseline","baseline_epoch80","use_query_bitmap_5","use_query_bitmap_10","use_query_bitmap_20","use_query_bitmap_50","use_query_bitmap_100","use_query_bitmap_200","use_query_bitmap_500"]
methods=["use_query_randombitmap"]
mods=["upd_heavy","ins_heavy","dist_shift","static"]
mskmp={"upd_heavy": [133,21,139,31,29,14,5,117,80,82],
       "ins_heavy": [42,32,116,40,115,22,66,9,58,125],
       "dist_shift": [118,43,101,82,136,24,65,81,18,138],
       "static":[48,119,58,121,142,47,139,69,105,50]
}

for mod in mods:
    # print("****************",mod,"****************")
    work_load_path="data/STATS/workload/"+mod+"/workload.sql"
    true_card_path="txt/query_true_card_"+mod+".txt"
    cards=np.load("data/STATS/workload/"+mod+"/features/all_cards.npy")
    test_id=np.load("data/STATS/workload/"+mod+"/features/test_sub_idxes.npy")
    tot=len(test_id)
    cards=cards[test_id]
    msk=get_important_idx(mod,work_load_path,true_card_path)
    for method in methods:
        # path="res/multiepoch/use_query_bitmap_0/epoch_"+str(idx)+"/e2e/ALECE_STATS_"+mod+".txt"
        # path="res/important/"+method+"/e2e/ALECE_STATS_"+mod+".txt"
        # if method=='pg':
        #     path="res/important/"+method+"/pg_STATS_"+mod+".txt"
        if mod!= "static":
            path="res/20240516/"+method+"/e2e/ALECE_STATS_"+mod.split("_")[0]+"_"+mod.split("_")[0]+".txt"
        else:
            path="res/20240516/"+method+"/e2e/ALECE_STATS_static.txt"
        # print(path)
        if not os.path.exists(path):
            continue
        print(method,end='\t')
        a=getnpy(path)
        qerror=np.maximum(cards/a,a/cards)
        idexes = np.where(qerror < 10)[0]
        n = idexes.shape[0]
        ratio =n / qerror.shape[0]
        qerror.sort()
        print(qerror[-int(tot*0.01)],end='\t')
        print(qerror[-int(tot*0.05)],end='\t')
        print(qerror[-int(tot*0.1)],end='\t')
        print(qerror[-int(tot*0.5)],end='\t')
        qerror=np.maximum(cards/a,a/cards)
        # qerror=qerror[msk]
        # print(a[msk[np.argmax(qerror)]],cards[msk[np.argmax(qerror)]])
        # qerror.sort()
        # print("msk:",end=' ')
        # for x in qerror[-10:]:
        #     print("{:.3f}".format(x),end=' ')
        # print()
        print(ratio)
# for mod in mods:
#     print("****************",mod,"****************")
#     work_load_path="data/STATS/workload/"+mod+"/workload.sql"
#     true_card_path="query_true_card_"+mod+".txt"
#     msk=get_important_idx(work_load_path,true_card_path)
#     path="res/baseline_epoch20_0/e2e/ALECE_STATS_"+mod+".txt"
#     a=getnpy(path)
#     a+=1
#     cards=np.load("data/STATS/workload/"+mod+"/features/all_cards.npy")
#     test_id=np.load("data/STATS/workload/"+mod+"/features/test_sub_idxes.npy")
#     cards=cards[test_id]
#     cards+=1
#     qerror=np.maximum(a/cards,cards/a)
#     # qerror=cards/a
#     # qerror=qerror[msk]
#     qerror.sort()
#     print(qerror[-20:])
