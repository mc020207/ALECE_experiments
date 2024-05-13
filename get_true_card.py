mods=["ins_heavy","upd_heavy","dist_shift","static"]
for mod in mods:
    print(mod)
    query_file=open("data/STATS/workload/"+mod+"/workload.sql","r")
    f=open(("optimal_single_tbls_STATS_"+mod.split('_')[0]+".txt"),"w")
    querys=query_file.readlines()
    for query in querys:
        if query.startswith("test_single_tbl"):
            f.write(query.split("||")[-2]+"\n")