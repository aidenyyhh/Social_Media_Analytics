"""
Social Media Analytics Project
Name: Aiden Che
"""

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]


'''
parseLabel(label)
Parameters: str
Returns: dict mapping str to str
'''
def parseLabel(label):
    result = {}
    splits1= label.replace("From: ", "").split(" (")
    result["name"] = splits1[0]
    splits2 = splits1[1].split(" from ")
    result["position"] = splits2[0]
    splits3= splits2[1].split(")")
    result["state"] = splits3[0]
    
    return result


'''
getRegionFromState(stateDf, state)
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    filtered_df = stateDf[stateDf["state"] == state]
    if not filtered_df.empty:
        region=filtered_df["region"].iloc[0]
    return region

'''
findHashtags(message)
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    hashtags = []
    i = 0
    while i < len(message):
        if message[i] == '#':
            hashtag = '#'
            i += 1
            while i < len(message) and message[i] not in endChars:
                hashtag += message[i]
                i += 1
            hashtags.append(hashtag)
        else:
            i += 1
    return hashtags



'''
findSentiment(classifier, message)
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    file = classifier.polarity_scores(message)
    message_Score = file["compound"]
    if message_Score > 0.1:
        return "positive"
    if message_Score < -0.1:
        return "negative"
    if -0.1<message_Score<0.1:
        return "neutral"

'''
addColumns(data, stateDf)
Parameters: dataframe ; dataframe
Returns: None
'''

def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []
    sentiments = []
    classifier = SentimentIntensityAnalyzer()

    for label in data['label']:
        
        label_info = parseLabel(label)
        names.append(label_info['name'])
        positions.append(label_info['position'])
        states.append(label_info['state'])
        region = getRegionFromState(stateDf, label_info['state'])
        regions.append(region)

    for text in data['text']:

        text_hashtags = findHashtags(text)
        hashtags.append(text_hashtags)
        
        sentiment_label = findSentiment(classifier, text)
        sentiments.append(sentiment_label)

    data['name'] = names
    data['position'] = positions
    data['state'] = states
    data['region'] = regions
    data['hashtags'] = hashtags
    data['sentiment'] = sentiments
    return None

'''
getDataCountByState(data, colName, dataToCount)
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    if colName == dataToCount == "":
        data = data
    else:
        data = data[data[colName] == dataToCount]
    StateCount = dict()
    for state in data["state"]:
        if state not in StateCount:
            StateCount[state] = 1
        else:
            StateCount[state] += 1
    return StateCount



'''
getDataForRegion(data, colName)
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    dic = {}
    for index, row in data.iterrows():
        Key = row["region"]
        Value = row[colName]
    
        if Key not in dic:
            dic[Key] = {}

        if Value in dic[Key]:
            dic[Key][Value] += 1
        else:
            dic[Key][Value] = 1
               
    return dic


'''
getHashtagRates(data)
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtags = {}
    tags = data["hashtags"]
    
    for tag in tags:
        if type(tag) == str:
            if tag not in hashtags:
                hashtags[tag] = 1
            else: 
                hashtags[tag] += 1
                
        elif type(tag) == list:
            for subtag in tag:
                if subtag not in hashtags:
                    hashtags[subtag] = 1
                else: 
                    hashtags[subtag] += 1
            
    return hashtags


'''
mostCommonHashtags(hashtags, count)
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    SearchDic = hashtags.copy()
    RankingDic = {}
    highest = 0
    
    while len(RankingDic) != count:
        for tags in SearchDic:
            if SearchDic[tags] > highest:
                Tag_of_highest = tags
                highest = SearchDic[tags]
                
        RankingDic[Tag_of_highest] = highest
        
        hashtags = SearchDic
        SearchDic.pop(Tag_of_highest)
        highest = 0

    return RankingDic


'''
getHashtagSentiment(data, hashtag)
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    message_Count = 0
    total_Sentiment_Score = 0
    
    for index, row in data.iterrows():
       if hashtag in row["hashtags"]:
           sentiment_Map = {"positive": 1, "negative": -1, "neutral": 0}
           total_Sentiment_Score += sentiment_Map[row["sentiment"]]
           message_Count += 1
           
    if message_Count == 0:
        return 0
    return total_Sentiment_Score/ message_Count

'''
graphStateCounts(stateCounts, title)
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    
    xLabel = []
    yLabel = []
    for key in stateCounts:
        xLabel.append(key)
        yLabel.append(stateCounts[key])
        
    plt.title(title)
    plt.bar(xLabel, yLabel)
    plt.xticks(rotation = "vertical")
    plt.show()
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''

def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    feature_rate_dic = {}
    for state, total_counts in stateCounts.items():
        if state in stateFeatureCounts:
            feature_rate_dic[state] = stateFeatureCounts[state]/total_counts
    
    ranking_dic = {}
       
    while len(ranking_dic) < n:
        highest = 0
        highest_state = None
        for state, rate in feature_rate_dic.items():
            if rate > highest:
                highest_state = state
                highest = rate
        if highest_state != None:
            ranking_dic[highest_state] = highest
            feature_rate_dic.pop(highest_state)
    
    graphStateCounts(ranking_dic, title) 

'''
graphRegionComparison(regionDicts, title)
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    feature_names = []
    region_names = []
    region_feature_list = []
    
    for region, feature in regionDicts.items():
        region_names.append(region)
        for feature_name in feature:
            if feature_name not in feature_names:
                feature_names.append(feature_name)
            
    for region in region_names:
        Temp_feature_value = []
        for feature_name in feature_names:
            if feature_name in regionDicts[region]:
                feature_count = regionDicts[region][feature_name]
            else:
                feature_count = 0
            Temp_feature_value.append(feature_count)
        region_feature_list.append(Temp_feature_value)

    xLabels = feature_names
    labelList = region_names
    valueLists = region_feature_list
    
    sideBySideBarPlots(xLabels, labelList, valueLists, title)
    
    return


'''
graphHashtagSentimentByFrequency(data)
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    New_dict= getHashtagRates(data)
    Top50Hashtags = mostCommonHashtags(New_dict, 50)
    
    hashtags = []
    frequencies = []
    sentiment_score = []
    
    for hashtag, count in Top50Hashtags.items():
        hashtags.append(hashtag)
        frequencies.append(count)
        sentiment_score.append(getHashtagSentiment(data,hashtag))
    
    scatterPlot(frequencies,sentiment_score, hashtags, "11")
    return


#### Provided Code by Professor Kelly Rivers in CMU ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()

