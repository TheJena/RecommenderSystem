---
author: Federico Motta
keywords:
- Amazon Reviews
- Information Retrival
- MongoDB
- SNAP
- mongodump
- mongorestore
subject: How to create test DB dump
---
\renewcommand*{\maketitle}{}
\hfill\textbf{\LARGE
    How to create "test" }\textsc{\LARGE db }\textbf{\LARGE dump}\hfill
\ \newline

Download the original Amazon reviews dataset from
[https://snap.stanford.edu/data/web-Amazon.html](https://snap.stanford.edu/data/web-Amazon.html)
and load it in a \textsc{MongoDB}.\newline

Otherwise clone it from an existing \textsc{MongoDB} instance
(depending on YourUniversity policies a \textsc{vpn} and a
\textsc{ssh} tunneling might be necessary):
```
$ sudo openfortivpn vpn.your_university.tld:443 --username ******** --password ********

$ ssh -fN -L 127117:localhost:21017 student**@descartes.departement.your_university.tld

$ sudo mongod    --config   /etc/mongodb.conf

$ mongodump      --host     127.0.0.1 --port     27117            \
                 --username ********  --password ********         \
                 --db       test      --archive           -j 8    \
  | mongorestore --host     127.0.0.1 --port     27017    -j 8
```
\ \newline
\ \newline
Then (inside the \textsc{MongoDB} shell) switch to "test" \textsc{db}
and since the dataset is really big, reduce its size by:

* dropping unneeded collections (like _restaurants_)
* cutting off from the "meta" and "reviews" collections the unneeded
  fields \newline
  (_brand_, _price_, _related_, _sales rank_, _title_, but also
  _helpful_, _review text_, _review time_, _summary_, _unix review time_)
* removing the documents without a "description" field or with an empty
  one
* populating an array with the "asin" fields of all the documents in the
  "meta" collection
* removing from the "reviews" collection documents about items with an
  "asin" field not present in the above mentioned array
\ \newline
```
$ mongo
> use test
> db.restaurants.drop()
> db.meta.update({},
                 { $unset: { brand:     1,
                             price:     1,
                             related:   1,
                             salesRank: 1,
                             title:     1,
                            },
                  },
                 { multi:true })
```
**Execution time**:\ \ \ \ \ \ \ \ \ \ \ \ `10.224 sec`\newline
**Updated documents**:\ \ \ \ `106`$\,$`474`\newline
\ \newline
```
> db.reviews.update({},
                    { $unset: { helpful:        1,
                                reviewText:     1,
                                reviewTime:     1,
                                summary:        1,
                                unixReviewTime: 1,
                               },
                     },
                    { multi:true })
```
**Execution time**:\ \ \ \ \ \ \ \ \ \ \ `19 min 21 sec`\newline
**Updated documents**:\ \ \ \ `23`$\,$`831`$\,$`908`\newline

```
> db.meta.deleteMany({description: {$exists: false}})
> db.meta.deleteMany({description: ''})
```
**Execution time**:\ \ \ \ \ \ \ \ \ \ \ \ `0.664 sec`\newline
**Updated documents**:\ \ \ \ `26`$\,$`264`\newline
\ \newline
```
> var items = db.meta.find({},
                           {_id:0, asin:1}).map(
                                   function(d) {return d.asin})
> db.reviews.deleteMany({asin: {$not: {$in: items}}})
```
**Execution time**:\ \ \ \ \ \ \ \ \ \ \ \ `10 min 2 sec`\newline
**Dropped documents**:\ \ \ \ `21`$\,$`867`$\,$`827`\newline

```
> db.meta.aggregate([{
        $addFields: {
                categories: {
                        $reduce: {
                                input: "$categories",
                                initialValue: [],
                                in: { $concatArrays: [
                                        "$$value",
                                        { $cond: {
                                                if: {$isArray: "$$this"},
                                                then: "$$this",
                                                else: []
                                                  }
                                         }
                                      ]
                                     }
                                  }
                             }
                     }
        },
        {$out: "meta"},
])
```
**Execution time**:\ \ \ \ \ `2.529 sec`\newline

\ \newline
Finally create an index on the "reviewerID" field in the "reviews"
collection and compact "meta" and "reviews" collections:
```
> db.reviews.createIndex({ reviewerID: 1 })
> db.runCommand ( { compact: "meta",    force: true } )
> db.runCommand ( { compact: "reviews", force: true } )
```
**Execution time**:\ \ \ \ `19.434 sec`\newline

\ \newline
Then restart \textsc{MongoDB} and create a dump of the shrinked "test" db:
```
$ mongodump --host 127.0.0.1 --port    27017                                         \
            --db   test      --archive=shrinked_test_db_at_descartes.mongodump.gz    \
            --gzip            -j       8

$ ls -sh    shrinked_test_db_at_descartes.mongodump.gz

  86M shrinked_test_db_at_descartes.mongodump.gz
```