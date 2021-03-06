Recommender System
======================

This project aims to do items recommendation in an e-commerce system.
Its major purpose is the "Gestione Avanzata dell'Informazione 2" exam
("Advanced Information Management 2" is a course about data analytics,
big data, NoSQL models, text analytics and graph analytics).

Start by having a look at the [presentation](https://github.com/TheJena/RecommenderSystem/raw/master/presentation.pdf).

The project is composed of several scripts to:

* inspect the available data in a MongoDB
* extract the needed data from a MongoDB
* apply a collaborative-filtering / collaborative-ranking approach to the data
* apply a content-based approach to the data
* merge the recommendations of the two approaches into a single list
  of item to recommend
* plot a performance comparison between the graph-based and the
  content-based approach

How to create MongoDB dump
=========================

Please take a look
[here](https://github.com/TheJena/RecommenderSystem/raw/master/docs/how_to_create_test_db_dump.pdf)
if you re interested in how the huge size of the "Amazon reviews"
dataset from [SNAP](https://snap.stanford.edu) has been reduced.

How to import MongoDB dump
==========================

First of all start a clean MongoDB instance; then in order to import
the compressed
[test](https://github.com/TheJena/RecommenderSystem/raw/master/dataset/shrinked_test_db.mongodump.gz)
db dump into your MongoDB you can simply run:

```
$ mongorestore --archive ./shrinked_test_db.mongodump.gz        \
$              --gzip                                           \
$              --objcheck                                       \
$              --verbose                                        \
$              -j 8
```

You can find a table showing the amount of items of each category in
the db [before](https://github.com/TheJena/RecommenderSystem/raw/master/docs/table_counting_items_in_each_category_before_db_shrinking.txt)
and [after](https://github.com/TheJena/RecommenderSystem/raw/master/docs/table_counting_items_in_each_category_after_db_shrinking.txt)
the original db reduction.

License
=======

Licensed under GPLv3+.
Full text available [here](https://github.com/TheJena/RecommenderSystem/raw/master/COPYING).