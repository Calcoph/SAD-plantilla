Si pone $ antes de un número, significa que extiende la descripción de esa prueba

|id|algoritmo|descripción|puntuación|puntuación (c negativa)|hiperparámetros
|:-|:-|:-|:-|:-|:-
1| GNB|tf-idf| 0.46164346705029863|0.7466583773702208| alpha=0.0000000001
2| GNB|$1 + eliminar con confidence < 0.6| 0.4365506247633813|0.7458048477315102|alpha=0.0000000001
3| GNB|$1 + $2 + emoji sentiment_score + borrar emojis + 10bins| 0.4365506247633813|0.7458048477315102|alpha=0.0000000001
4| GNB|$1 + $2 + emoji sentiment_score + 10bins| 0.4365506247633813|0.7458048477315102|alpha=0.0000000001
5| GNB|$1 + $2 + emoji positive+negative+neutral + borrar emojis + 10bins| 0.4357424172767395|0.7451102142191864|alpha=0.0000000001
6| GNB|$1 + $2 + emoji positive+negative+neutral + 10bins| 0.4357424172767395|0.7451102142191864|alpha=0.0000000001
7| GNB|$1 + $2 + emoji positive+negative+neutral+sentiment_score + borrar emojis + 10bins| 0.4357424172767395|0.7451102142191864|alpha=0.0000000001
8| GNB|$1 + $2 + emoji positive+negative+neutral+sentiment_score + borrar emojis + 10bins| 0.4357424172767395|0.7451102142191864|alpha=0.0000000001
9| GNB|$3+100bins| 0.4365506247633813|0.7458048477315102|alpha=0.0000000001
10| GNB|$3+1000bins| 0.4365506247633813|0.7458048477315102|alpha=0.0000000001
11| MNB|$1| 0.2531502375143104|0.7557053941908713|alpha=1.0
12| MNB|$3| 0.25330879130511|0.75992637391533|alpha=1.0
13| MNB|$9| 0.25330879130511|0.75992637391533|alpha=1.0
14| CNB|$1| 0.25183648777115203|0.755509463313456|alpha=1.0
15| CNB|$8| ERROR | ERROR
16| RF|tf-idf (depth 4000-5000, samples 99-100)|0.4771907257199133|0.7357615894039735|max_depth=4000, min_samples_leaf=99
17| GNB|$1+eliminar con confidence < 0.5| 0.4369940954524291 |0.7506172839506173|alpha=0.0000000001
18| GNB|$1+eliminar con confidence < 0.7| 0.40763768270931594 |0.8165765765765766|alpha=0.0000000001
19| GNB|$1+eliminar con confidence < 0.4| 0.4369940954524291|0.7506172839506173|alpha=0.0000000001
20| GNB|$1+eliminar con confidence > 0.7| 0.41883195655606015|0.548951048951049|alpha=0.0000000001
21| RF|tf-idf (depth 2-30, samples 10-50)| 0.6477630177614143|0.8170689052116126|max_depth=(17, 20), min_samples_leaf=10
22| RF|tf-idf (depth 2-30, samples 10-50)+emoji sentiment_score + borrar_emojis| 0.6714055772940525|0.8170988086895585|max_depth=(22, 25), min_samples_leaf=10
23| RF|tf-idf (depth 1000, samples 5-1000)| 0.6659335437685632|0.832535885167464|max_depth=1000, min_samples_leaf=5
24| RF|tf-idf (depth 20-30, samples 50)| 0.5013711298857654|0.7458699472759227|max_depth=20, min_samples_leaf=50
25| RF|tf-idf (depth 10-30, samples 1-10)|0.6690424357483918|0.8361809045226131|max_depth=(14, 30), min_samples_leaf=(4, 2)
26| RF|tf-idf (depth 13-23, samples 3-7)|0.6690424357483918|0.8257756563245824|max_depth=(14, 23), min_samples_leaf=(4, 3)
27| RF | $26 + emoji sentiment_score + borrar_emojis|0.6766391654184885|0.8250779355732595|max_depth=(17, 22), min_samples_leaf=(4, 3)
28| RF | $26 + emoji sentiment_score|0.6832097718888496|0.831759356418328|max_depth=15, min_samples_leaf=5
29| RF | $26 + emoji positive|0.6778098450304378|0.8290807409996505|max_depth=(22, 21), min_samples_leaf=(3, 5)
30| RF | $26 + emoji positive+neutral+negative|0.6747283092303015|0.8307372793354101|max_depth=(14, 21), min_samples_leaf=(3, 5)
31| RF | $26 + emoji positive+neutral+negative+sentiment_score|0.6794446334997019|0.8293871866295265|max_depth=(21,23), min_samples_leaf=(3,4)
32| RF | $28 +  borrar emojis|0.6766391654184885|0.8250779355732595|max_depth=(17, 22), min_samples_leaf=(4, 3)
33| RF | $27 + lematizar | 0.6850769714351029 |-| max_depth=23, min_samples_leaf=5
34|RF|$27-binning|0.6775537649770463|0.8231221876081689|max_depth=22, min_samples_leaf=3
35| CNB| tf-idf + (alpha 0.01-3.0 step=0.05)| ERROR | ERROR
36|MNB|tf-idf+bin tfidf + (alpha 0.01-3.0 step=0.05)|0.5599917771833144|0.8087793144918822|alpha=0.01
38|CNB|$1+bin tfidf|0.46164346705029863|0.755509463313456|alpha=1
39|RF|$28+bin tfidf|0.6860607281211802|0.8312958435207825|max_depth=(13, 21), min_samples_leaf=(4, 5)
40|MNB|tf-idf+bin tfidf + (alpha 0.001-0.01 step=0.005)|0.5628223913483483|0.8112183353437877|alpha=0.006
41|MNB|tf-idf+bin tfidf + (alpha 0.001-0.01 step=0.0005)|0.5628223913483483|0.8112183353437877|alpha=0.006
42|GNB|tf-idf+bin tfidf + (alpha 0.001-0.01 step=0.005)|0.48572906975322444|0.7384615384615385|alpha=(0.006, 0.001)
43|GNB|tf-idf+bin tfidf + (alpha 0.001-0.01 step=0.0005)|0.4918555075909052|0.7384615384615385|alpha=(0.025, 0.01)
44|RF|$39+undersampling|0.6562686824923458|0.6625463535228677|max_depth=(22, 19), min_samples_leaf=3
45|RF|$39+oversampling|0.6781939017964169|0.7142857142857143|max_depth=(16, 20), min_samples_leaf=(6, 7)
46|MNB|$41+undersampling|0.5924689759863742|0.5552884615384616|alpha=0.095
47|MNB|$41+oversampling|0.5363993884316556|0.6247008137865007|alpha=0.095

con y sin strip_accents -> no cambia nada

con y sin stop words -> Elegidas a mano: no cambia nada. Usando inglés -> empeora
