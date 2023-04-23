Si pone $ antes de un número, significa que extiende la descripción de esa prueba

|id|algoritmo|descripción|puntuación|hiperparámetros
|:-|:-|:-|:-|:-|
1| gaussian naive bayes|tf-idf| 0.46164346705029863
2| gaussian naive bayes|$1 + eliminar con confidence < 0.6| 0.4365506247633813
3| gaussian naive bayes|$1 + $2 + emoji sentiment_score + borrar emojis + 10bins| 0.4365506247633813
4| gaussian naive bayes|$1 + $2 + emoji sentiment_score + 10bins| 0.4365506247633813
5| gaussian naive bayes|$1 + $2 + emoji positive+negative+neutral + borrar emojis + 10bins| 0.4357424172767395
6| gaussian naive bayes|$1 + $2 + emoji positive+negative+neutral + 10bins| 0.4357424172767395
7| gaussian naive bayes|$1 + $2 + emoji positive+negative+neutral+sentiment_score + borrar emojis + 10bins| 0.4357424172767395
8| gaussian naive bayes|$1 + $2 + emoji positive+negative+neutral+sentiment_score + borrar emojis + 10bins| 0.4357424172767395
9| gaussian naive bayes|$3+100bins| 0.4365506247633813
10| gaussian naive bayes|$3+1000bins| 0.4365506247633813
11| multinomial naive bayes|$1+multinomial| 0.2531502375143104
12| multinomial naive bayes|$3+multinomial| 0.25330879130511
13| multinomial naive bayes|$9+multinomial| 0.25330879130511
14| categorical naive bayes|$1+categorical| 0.25183648777115203
15| categorical naive bayes|$8+categorical| ERROR
16| random forest|tf-idf (depth 4000-5000, samples 3-7)|0.4771907257199133|max_depth=4000, min_samples_leaf=99
17| gaussian naive bayes|$1+eliminar con confidence < 0.5| 0.4369940954524291
18| gaussian naive bayes|$1+eliminar con confidence < 0.7| 0.40763768270931594
19| gaussian naive bayes|$1+eliminar con confidence < 0.4| 0.4369940954524291
20| gaussian naive bayes|$1+eliminar con confidence > 0.7| 0.41883195655606015
21| random forest|tf-idf (depth 2-30, samples 10-50)| 0.6477630177614143|max_depth=17, min_samples_leaf=10
22| random forest|tf-idf (depth 2-30, samples 10-50)+emoji sentiment_score + borrar_emojis| 0.6714055772940525|max_depth=22, min_samples_leaf=10
23| random forest|tf-idf (depth 1000, samples 5-1000)| 0.6659335437685632|max_depth=1000, min_samples_leaf=5
24| random forest|tf-idf (depth 20-30, samples 50)| 0.5013711298857654|max_depth=20, min_samples_leaf=50
25| random forest|tf-idf (depth 10-30, samples 1-10)|0.6690424357483918|max_depth=14, min_samples_leaf=4
26| random forest|tf-idf (depth 13-23, samples 3-7)|0.6690424357483918|max_depth=14, min_samples_leaf=4
27| random forest | $26 + emoji sentiment_score + borrar_emojis|0.6766391654184885|max_depth=17, min_samples_leaf=4
28| random forest | $26 + emoji sentiment_score|0.6832097718888496|max_depth=15, min_samples_leaf=5
29| random forest | $26 + emoji positive|0.6778098450304378|max_depth=22, min_samples_leaf=3
30| random forest | $26 + emoji positive+neutral+negative|0.6747283092303015|max_depth=14, min_samples_leaf=3
31| random forest | $26 + emoji positive+neutral+negative+sentiment_score|0.6794446334997019|max_depth=21, min_samples_leaf=3