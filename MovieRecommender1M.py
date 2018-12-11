#this file is executed locally

import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def loadMovieNames():
    #this function will create an rdd which contains a dictionary of (movieID, movieName)
    # this will be used as a look up table later when we get the movie id's
    
    movieNames = {}
    with open("ml-1m/movies.dat") as f:
        for row in f:
            columns = row.split("::")
            movieNames[int(columns[0])] = columns[1].decode('ascii', 'ignore')
    return movieNames

def makePairs((user, ratings)):
    (a,b) = ratings[0]
    (c, d) = ratings[1]
    (m1, r1) = (a,b)
    (m2, r2) = (c,d)
    return ((m1, m2), (r1, r2))

def removeDuplicates( (userID, ratings) ):
    (a,b) = ratings[0]
    (c, d) = ratings[1]
    (m1, r1) = (a,b)
    (m2, r2) = (c,d)
    return m1 < m2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


#local[*] will allow spark to tell the os that it needs every core of every cpu of the computer.
conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
#Creating Spark Context from the configuration above.
sparkCtx = SparkContext(conf = conf)

print("\nLoading movie names...")
movieLookUp = loadMovieNames()

data = sparkCtx.textFile("ml-1m/ratings.dat")

ratings = data.map(lambda l: l.split("::")).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

ratingsPartitioned = ratings.partitionBy(100)
joinedRatings = ratingsPartitioned.join(ratingsPartitioned)

uniqueJoinedRatings = joinedRatings.filter(removeDuplicates)

moviePairs = uniqueJoinedRatings.map(makePairs).partitionBy(100)

moviePairRatings = moviePairs.groupByKey()

moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).persist()

if (len(sys.argv) > 1):

    scoreThreshold = 0.97
    coOccurenceThreshold = 1000

    movieID = int(sys.argv[1])

    filteredResults = moviePairSimilarities.filter(lambda((pair,sim)): \
        (pair[0] == movieID or pair[1] == movieID) \
        and sim[0] > scoreThreshold and sim[1] > coOccurenceThreshold)

    results = filteredResults.map(lambda((pair,sim)): (sim, pair)).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + movieLookUp[movieID])
    for result in results:
        (sim, pair) = result
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(movieLookUp[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
