import numpy as np
from tqdm import tqdm
def check2(data,attr_range):
    # print(data)
    # print(attr_range)
    assert(len(data)*2==len(attr_range))
    for i in range(len(data)):
        if not (data[i]<=attr_range[2*i+1] and data[i]>=attr_range[2*i]):
            return 0
    return 1

def check(sample,attr_range):
    ans=[]
    for data in sample:
        ans.append(check2(data,attr_range))
    return ans


sample_nums=[5,10,20,50,100,200,500]

for sample_num in sample_nums:
    np.random.seed(998244353)
    tables=['badges','comments','posthistory','postlinks','posts','tags','users','votes']
    mods=["upd_heavy","ins_heavy","dist_shift","static"]
    # sample_num=200
    samples=[]
    table_num_attr=[]
    for i,table in enumerate(tables):
        csv_file=open("../data/STATS/data/"+table+".csv")
        lines=csv_file.readlines()
        table_num_attr.append(len(lines[0].split(',')))
        tot_num=len(lines)
        msk=np.arange(1,tot_num)
        np.random.shuffle(msk)
        msk=msk[:sample_num]
        sample=[]
        for idx in msk:
            line=lines[idx]
            terms=line.split(',')
            for i in range(len(terms)):
                if terms[i][-1]=="\n":
                    terms[i]=terms[i][:-1]
                terms[i]=int(terms[i])
            sample.append(terms)
        samples.append(sample)
        # print(sample)
    for mod in mods:
        meta_infos=np.load("../data/STATS/workload/"+mod+"/features/feature_meta_infos.npy")
        all_features=np.load("../data/STATS/workload/"+mod+"/features/all_features.npy")
        attr_range_conds_list=np.load("../data/STATS/workload/"+mod+"/features/attr_range_conds_list.npy")
        [histogram_feature_dim, query_feature_dim, num_attrs, n_possible_joins] = meta_infos
        # print(table_num_attr)
        table_features = all_features[:, histogram_feature_dim:histogram_feature_dim+len(tables)]
        ans=np.zeros(shape=(all_features.shape[0],sample_num*len(tables)))
        for i in tqdm(range(all_features.shape[0])):
            table_feature=table_features[i]
            nowidx=0
            # print(table_feature)
            for j in range(len(tables)):
                if table_feature[j]:
                    ans[i][j*sample_num:(j+1)*sample_num]=check(samples[j],attr_range_conds_list[i][2*nowidx:2*(nowidx+table_num_attr[j])])
                nowidx+=table_num_attr[j]
        np.save("../data/STATS/workload/"+mod+"/features/bitmap_"+str(sample_num)+".npy",ans)