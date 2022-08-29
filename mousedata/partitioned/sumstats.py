import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

bb = []
mm = []
for i in range(20):
    bb.append(pd.read_csv("bb" + str(i+1) + ".csv"))
    mm.append(pd.read_csv("bb" + str(i+1) + ".csv"))
    
bbstats = []
mmstats = []

for period in bb:
    stats = {}
    stats["aps"] = float(max(period["ElapsedTime"])) / float(len(period["ElapsedTime"]))
    tot_distance = 0
    no_moves = 1
    #key-mouse gap
    kmgap = 0
    gapcount = 0
    counter = 0
    for i in range(len(period["ElapsedTime"]) - 1):
        if period["Action"][i] == "Moved":
            if period["Action"][i+1] == "Moved":
                tot_distance += float((np.sqrt(np.square(period["X"][i+1]-period["X"][i]) + np.square(period["Y"][i+1]-period["Y"][i]))))
            else:
                pass
            no_moves += 1
            if counter > 0:
                kmgap += (period["ElapsedTime"][i] - counter)
                gapcount += 1
            counter = 0
        else:
            counter = period["ElapsedTime"][i]
    try:
        float(float(tot_distance) / float(no_moves))
        stats["speed"] = float(tot_distance) / float(no_moves)
    except:
        stats["speed"] = 0.0
    stats["speed"] = float(tot_distance) / float(no_moves)
    stats["xvar"] = np.var(period["X"])
    stats["yvar"] = np.var(period["Y"])
    if gapcount > 0:
        stats["key_mouse_gap"] = float(kmgap) / float(gapcount)
    else:
        stats["key_mouse_gap"] = 0.0
    bbstats.append(stats)
    
    
for period in mm:
    stats = {}
    stats["aps"] = float(max(period["ElapsedTime"])) / float(len(period["ElapsedTime"]))
    tot_distance = 0
    no_moves = 1
    #key-mouse gap
    kmgap = 0
    gapcount = 0
    counter = 0
    for i in range(len(period["ElapsedTime"]) - 1):
        if period["Action"][i] == "Moved":
            if period["Action"][i+1] == "Moved":
                tot_distance += float((np.sqrt(np.square(period["X"][i+1]-period["X"][i]) + np.square(period["Y"][i+1]-period["Y"][i]))))
            else:
                pass
            no_moves += 1
            if counter > 0:
                kmgap += (period["ElapsedTime"][i] - counter)
                gapcount += 1
            counter = 0
        else:
            counter = period["ElapsedTime"][i]
    stats["speed"] = float(tot_distance) / float(no_moves)
    stats["xvar"] = np.var(period["X"])
    stats["yvar"] = np.var(period["Y"])
    if gapcount > 0:
        stats["key_mouse_gap"] = float(kmgap) / float(gapcount)
    else:
        stats["key_mouse_gap"] = 0.0
    mmstats.append(stats)

combined = bbstats + mmstats
c_format = []

for dic in combined:
    c_format.append(np.fromiter(dic.values(), dtype=float))

print(KMeans(n_clusters=2, random_state=0).fit_predict(c_format))
#print(kmeans.labels_)
#print(len(c_format))