from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from __future__ import division
import string
import math
import csv
import random
import time
import pandas as pd

# GLOBALS:
BUDGET = 6250000


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
            instance=process_event(row,STD_SLOTPRICE)
            data[row[24]].append(instance)
            labels[row[24]].append(int(row[0]))
    print "data and labels loaded"
    return data,labels


def trainCTRmodel(training_events, training_labels):
    models = {}
    for key in training_events.keys():
        data=training_events[key]
        labels= training_labels[key]


        label_encoder = LabelEncoder()
        vectorizer = DictVectorizer(sparse=True)

        train_event_x = vectorizer.fit_transform(data)
        train_event_y = label_encoder.fit_transform(labels)

        # Getting the class weight to rebalance data:
        #neg_weight = sum(labels) / len(labels)
        #pos_weight = 1 - neg_weight

        # Create and train the model.
        #p = 0.34
        #lr = LogisticRegression(C=p, class_weight={1: pos_weight, 0: neg_weight})
        #lr=LogisticRegressionCV(class_weight={1: pos_weight, 0: neg_weight})
        lr = SGDClassifier(class_weight="balanced",learning_rate='optimal', \
                           n_iter=20,loss="log",penalty='elasticnet')
        #print "labels: "+str(labels)
        lr.fit(train_event_x, train_event_y)
        models[key] = (lr, label_encoder, vectorizer)
        print('Training model for advertiser %s done')%(key)
    return models


def process_event(row,STD_SLOTPRICE,data_type="Training"):
    # Initilize instance:
    if data_type=="Training":
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17], 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}
        instance = update_usertag_os_browser(instance,row)
        return instance

    elif data_type=="Validation":
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17],'payprice':int(row[22]), 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}
        instance = update_usertag_os_browser(instance, row)
        return instance
    else:
        instance = {'weekday': row[0], 'hour': row[1], 'region': row[7], \
                    'city': row[8], 'adexchange': row[9], 'slotwidth': row[14], 'slotheight': row[15], \
                    'slotvisibility': row[16],'slotformat': row[17], 'slotprice': float(row[18]) / STD_SLOTPRICE, \
                    'advertiser': row[21]}
        instance = update_usertag_os_browser(instance, row,col_usertag=22,col_os_brow=5)
        return instance


def update_usertag_os_browser(instance,row,col_usertag=25,col_os_brow=6):
    usertags = row[col_usertag].split(',')
    temp_dict = {}
    for tag in usertags:
        temp_dict["tag " + tag] = True
    instance.update(temp_dict)
    # add OS and browser:
    op_sys, browser = row[col_os_brow].split('_')
    instance.update({op_sys: True, browser: True})
    return instance


def predict_event_CTR(instance,advertiser, model): # models:dict
    #print "model adv " +str(model[advertiser])
    lr = model[advertiser][0]
    # Transform event:
    label_encoder = model[advertiser][1]
    vectorizer = model[advertiser][2]
    event = [instance]
    event_x = vectorizer.transform(event)
    #event_y = label_encoder.inverse_transform(lr.predict(event_x))
    event_y = lr.predict_proba(event_x)
    return event_y[0][1]


def RTB_simulation(model, validation_path, training_path,
                   start_budget = BUDGET, lambda_const=5.2e-7, c=80, data_type="Validation"):  # param is the dictionary with the bidprice per advertiser
    impressions = 0
    clicks = 0
    spend = 0
    budget=start_budget
    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)
    results=[]

    # Linear Stragegy:
    avgCTR,base_bid = get_LRS_params(training_path)


    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            advertiser=row[24]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, data_type = data_type)

            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, model)
            #print "pCTR: "+str(pCTR)
            # Calculate the bid based on ORTB:
            #current_bid = np.sqrt((c*pCTR)/lambda_const+np.power(c,2))-c

            # Second way:
            current_bid=np.power((pCTR+np.square((c**2*lambda_const**2+pCTR**2))/(c*lambda_const)),(1/3))- \
                                np.power(((c*lambda_const)/(pCTR+np.square((c**2*lambda_const**2+pCTR**2)))),(1/3))

            # Check if we still have budget:
            if budget > current_bid:

                # Get the market price:
                payprice = instance['payprice']

                # Check if we win the bid:
                if current_bid > payprice:
                    impressions += 1
                    budget -= payprice
                    # Check if the person clicks:
                    if row[0] == "1":
                        print "current bid : %d , payprice: %d, click? : %s" % (int(current_bid), int(payprice), row[0])
                        clicks += 1
            results.append((row[3],advertiser,current_bid,payprice,int(row[22]),row[1]))

    print("Impressions:{0}".format(impressions))
    print("Clicks:{0}".format(clicks))
    print("Reamining Budget:{0}".format(budget))
    if impressions > 0:
        #print "Best bid CTR: " + str((clicks / impressions) * 100)
        finalCTR=(clicks / impressions) * 100
        return finalCTR,results
    else:
        return -1,results



def RTB_simulation_test(model, validation_path, start_budget = BUDGET,
                        lambda_const=5.2e-7, c=80, data_type="test"):  # param is the dictionary with the bidprice per advertiser

    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)
    print "got std "
    results=[]

    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            advertiser=row[21]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, data_type = data_type)

            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, model)

            # Calculate the bid based on ORTB:
            current_bid = np.sqrt((c*pCTR)/lambda_const+np.power(c,2))-c

            # Second way:
            #current_bid=np.power((pCTR+np.square((c**2*lambda_const**2+pCTR**2))/(c*lambda_const)),(1/3))- \
            #                    np.power(((c*lambda_const)/(pCTR+np.square((c**2*lambda_const**2+pCTR**2)))),(1/3))
            results.append((row[2],current_bid))

        return results


def find_optimal_c(model, validation_path, training_path):  # param is the dictionary with the bidprice per advertiser:
    c_range=np.arange(20,200,15)
    val_c_CTR=[]
    for c in c_range:
        predCTR=RTB_simulation(model, validation_path, training_path,c=c)
        val_c_CTR.append((c,predCTR))
        print "c: %d, CTR: %0.5f"%(c,predCTR)
    return val_c_CTR


def find_optimal_lambda(model, validation_path, training_path):  # param is the dictionary with the bidprice per advertiser:
    lambda_range=np.logspace(-7,1,10)
    val_c_CTR=[]
    for lamb in lambda_range:
        predCTR=RTB_simulation(model, validation_path, training_path, lambda_const=lamb, c=140)
        val_c_CTR.append((lamb,predCTR))
        print "lambda: %d, CTR: %0.5f"%(lamb,predCTR)
    return val_c_CTR


def find_optimal_params(model, validation_path, training_path):  # param is the dictionary with the bidprice per advertiser:
    c_range = np.arange(20, 200, 15)
    lambda_range=np.logspace(-7,1,10)
    params_CTR=[]
    for lamb in lambda_range:
        for c in c_range:
            predCTR=RTB_simulation(model, validation_path, training_path, lambda_const=lamb, c=c)
            params_CTR.append((c,lamb,predCTR))
            print "lambda: %0.2f, c: %d, CTR: %0.5f"%(lamb,c,predCTR)
    return params_CTR


def results_to_csv(results):
    results=np.array(results)
    results=pd.DataFrame(results,columns=["bidid","bidpice"])
    return results





if __name__=="__main__":
    # MAIN:
    st=time.time()
    training_path = r"../dataset/train.csv"
    validation_path = r"../dataset/validation.csv"
    test_path = r"../dataset/test.csv"

    # Extracting data:
    training_events_best, labels_best = load_data(training_path)

    # training model
    models_best_CTR= trainCTRmodel(training_events_best, labels_best)

    val_best_CTR,results=RTB_simulation(models_best_CTR, validation_path, training_path,c=90 ,data_type="Validation")

    #optimal_c=find_optimal_c(models_best_CTR,validation_path,training_path)
    # optimal_lambda = find_optimal_lambda(models_best_CTR, validation_path, training_path)
    #optimal_params = find_optimal_params(models_best_CTR, validation_path, training_path)


    #results_test = RTB_simulation_test(models_best_CTR, test_path, c=60,lambda_const=5.2e-3,data_type="Test")

    #optimal_c=find_optimal_c(models_best_CTR,validation_path,training_path)
    # optimal_lambda = find_optimal_lambda(models_best_CTR, validation_path, training_path)
    #optimal_params = find_optimal_params(models_best_CTR, validation_path, training_path)

    #results_df=results_to_csv(results_test)