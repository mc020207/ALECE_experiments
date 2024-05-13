path="data/STATS/workload/upd_heavy/workload.sql"
f=open(path,"r")
lines=f.readlines()
mp={'insert':0,'update':0,'delete':0,'train_query':0,'train_sub_query':0,'test_query':0,'test_sub_query':0}
for line in lines:
    for key in mp.keys():
        if line.startswith(key):
            mp[key]+=1
print(mp)
            