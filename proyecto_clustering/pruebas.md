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
16: barrido de hiperparámetro num docs (3-50, step=3) -> alrededor de 10 parece lo mejor
17: barrido de hiperparámetro no_above (0.05-0.9, step=0.05) -> parece random, dejar en 0.2
18: barrido alpha -> no se aprecia diferencia
19: barrido beta -> no se aprecia diferencia
20: barrido chunksize -> 2500
21: barrido iterations -> no se aprecia diferencia
22: barrido passes -> no se aprecia diferencia (espera a que acabe y luego lo confirmas)
23: difference score barrido iterations -> no se aprecia diferencia
24: $15+quitar http,co,amp,https
