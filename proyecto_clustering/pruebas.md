1: eliminar las de por encima de 50% de documentos
2: eliminar las de por encima de 20% de documentos
3: eliminar las de por encima de 20% de documentos
4: 10% docs, solo negativas, de united, cambiando stopwords a stopwords = STOPWORDS.union(frozenset(["ua", "we", "on", "me", "you"])).difference(["kg", "before", "over", "system", "serious", "off", "take", "nowhere", "found"]), quitar puntuación
5: $4+barrido de k [2,30)
6: $4+12topics
7: $5+positivos
8: $5+min5docs
9: $8+90%
10: $8+50 passes
11: $6+
12: $11+alpha=0.01
13: $11+alpha=0.1
14: $11+alpha=0.5
14: $11+alpha=0.0001
