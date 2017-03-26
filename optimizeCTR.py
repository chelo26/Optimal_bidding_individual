from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from __future__ import division
import string
import math
import csv
import random
import time
import pandas as pd


def get_std_slotprice(path,column="slotprice"):
    df = pd.read_csv(path, skipinitialspace=True, usecols=[column])
    return int(df.slotprice.values.std())

def get_LRS_params(path):
    df=pd.read_csv(path)
    avgCTR=(df.click.sum()/df.shape[0])*100
    base_bid=df.payprice.mean()
    return avgCTR,base_bid

def load_data(filepath,training=True):
    data = defaultdict(list)
    labels = defaultdict(list)
    # std of slotprice for nomalization
    STD_SLOTPRICE = get_std_slotprice(filepath,column="slotprice")
    print "std stored"
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # pass header:
        next(reader)
        # Iterate:
        for row in reader:
            instance=process_event(row,STD_SLOTPRICE,training)
            data[row[24]].append(instance)
            labels[row[24]].append(int(row[0]))
    print "data and labels loaded"
    return data,labels


def trainCTRmodel(training_events, training_labels):
    models = {}
    for key in training_events.keys():
        data=training_events[key]
        labels= training_labels[key]

        # Encoders:
        label_encoder = LabelEncoder()
        vectorizer = DictVectorizer()

        # Vectorizing:
        train_event_x = vectorizer.fit_transform(data)
        train_event_y = label_encoder.fit_transform(labels)

        # Getting the class weight to rebalance data:
        #neg_weight = sum(labels) / len(labels)
        #pos_weight = 1 - neg_weight

        # Create and train the model -----> IMPORTANT:
        #p = 0.34
        #model = LogisticRegression(C=p, class_weight={1: pos_weight, 0: neg_weight})
        #model = LogisticRegressionCV(class_weight={1: pos_weight, 0: neg_weight})

        model = SGDClassifier(class_weight="balanced",learning_rate='optimal', \
                                n_iter=20,loss="log",penalty='elasticnet')
        #print "labels: "+str(labels)
        #model = RandomForestClassifier(n_estimators=120)
        model.fit(train_event_x, train_event_y)
        models[key] = (model, label_encoder, vectorizer)
        print('Training model for advertiser %s done')%(key)
    return models,data


def process_event(row,STD_SLOTPRICE,training=True):
    # Initilize instance:
    if training==True:
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17], 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}
    else:
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17],'payprice':int(row[22]), 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}

    # Add usertags:
    usertags = row[25].split(',')
    temp_dict = {}
    for tag in usertags:
        temp_dict["tag " + tag] = True
    instance.update(temp_dict)
    # add OS and browser:
    op_sys, browser = row[6].split('_')
    instance.update({op_sys: True, browser: True})
    return instance

def predict_event_CTR(instance,advertiser, model): # models:dict
    #print "model adv " +str(model[advertiser])
    lr = model[advertiser][0]
    # Transform event:
    vectorizer = model[advertiser][2]
    event = [instance]
    event_x = vectorizer.transform(event)
    event_y = lr.predict_proba(event_x)
    return event_y[0][1]

def testCTRmodel(models_dictio,validation_path):
    performance=defaultdict(lambda : defaultdict(float))
    y_pred=[]
    y_test=[]
    adv_y_pred = defaultdict(list)
    adv_y_true = defaultdict(list)
    impressions = 0
    clicks = 0

    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)

    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)

        for row in reader:
            advertiser=row[24]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, training=False)
            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, models_dictio)
            true_click = int(row[0])
            y_test.append(true_click)
            pred_click= int(pCTR>=0.5)
            y_pred.append(pred_click)

            # adding to the dictio:
            adv_y_pred[advertiser].append(pred_click)
            adv_y_true[advertiser].append(true_click)


            if pCTR > 0.75:
                impressions += 1
                performance[advertiser]['impressions'] += 1
                if true_click == 1:
                    clicks += 1
                    performance[advertiser]['clicks'] += 1

            #performance[advertiser]['impressions']+=1

            if performance[advertiser]['impressions']>0:
                performance[advertiser]['CTR'] = performance[advertiser]['clicks']/performance[advertiser]['impressions']


    print("Impressions:{0}".format(impressions))
    print("Clicks:{0}".format(clicks))

    return performance, y_test, y_pred,adv_y_true,adv_y_pred
    # finalCTR= (clicks / impressions) * 100
    # if impressions > 0:
    #     print "Best bid CTR: " + str((clicks / impressions) * 100)
    #     return finalCTR,clicks,impressions
    # else:
    #     return -1,clicks,impressions


def results_to_csv(results):
    results=np.array(results)
    results=pd.DataFrame(results,columns=["bidid","bidprice"])
    return results



if __name__=="__main__":
    # MAIN:
    st=time.time()
    training_path = r"../dataset/train.csv"
    validation_path = r"../dataset/validation.csv"

    # Extracting data:
    training_events, training_labels = load_data(training_path)
    validation_events,validation_labels = load_data(validation_path)


    # training model
    st = time.time()
    CTR_model,data= trainCTRmodel(training_events, training_labels)

    perf,y_true,y_pred,adv_true,adv_pred =testCTRmodel(CTR_model, validation_path)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    print auc


    adv_auc={}
    for key in adv_pred.keys():
        fpr, tpr, thresholds = metrics.roc_curve(adv_true[key], adv_pred[key])
        auc = metrics.auc(fpr, tpr)
        adv_auc[key]=auc
        print ("adv: %s, auc: %0.3f" %(key,auc))


    print time.time()-st
