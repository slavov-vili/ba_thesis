from bunch import Bunch
import datetime
import matplotlib.pyplot as plt
import numpy             as np
import random



model_params = {"alpha_d":   0.3,   # default alpha for all items
                "alpha_min": 0.1,   # minimum possible alpha
                "alpha_max": 0.5,   # maximum possible alpha
                "c":         0.21,  # decay scaling factor
                "tau":      -0.8,   # activation threshold
                "s":         0.255} # recall probability noise reduction factor



# list all items to be learned
items = ["noodles", "where", "many", "way", "market", "castle", "group", "restaurant", "amazing", "to visit", "each", "tree", "British", "adult", "a day", "open(from...to...)", "furniture", "a year", "open", "free time", "canal", "Chinese", "stall", "playing field", "fancy", "a week", "to enjoy", "best", "wonderful", "expensive", "to add", "boat", "to join in", "view", "canoeing", "flower", "area"] # end items list



# maps each item to its values
items_info = {items[0]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[1]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[2]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[3]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[4]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[5]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[6]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[7]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[8]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[9]:  Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[10]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[11]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[12]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[13]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[14]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[15]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[16]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[17]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[18]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[19]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[20]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[21]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[22]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[23]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[24]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[25]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[26]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[27]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[28]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[29]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[30]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[31]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[32]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[33]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[34]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[35]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0),
              items[36]: Bunch(alpha=model_params["alpha_d"], encounters=[], incorrect=0)}  # end item map





def learn(items, items_info):
    """
    Simulates learning process by adding new encounters of words.
    """
    
    # TODO: 2 sessions, separated by 24 hours
    # TODO: each session lasts 30 mins
    # TODO: each encounter lasts between 0 and 15 seconds

    # store time of the current encounter
    cur_time = datetime.datetime.now()
    # add 15 seconds to the current time (useful when checking if below threshold)
    cur_time = cur_time.timedelta(seconds=15)
    # the index of the next NEW item which needs to be learned
    next_new_item_idx = 0

    # get the next item to be presented
    item, item_act, next_new_item_idx = get_next_item(items, next_new_item_idx, items_info, cur_time)
    # calculate the item's recall probability
    item_rec = calc_recall_prob(item_act)
    # try to guess the item
    guessed = guess_item(item_rec)

    # adjust values depending on outcome
    if guessed:
        items_info[item].alpha -= 0.001
    else:
        items_info[item].alpha += 0.001
        items_info[item].incorrect++

    # add the current encounter to the item's encounter list
    items_info[item].encounters.append(cur_time)
    return



def get_next_item(items, next_new_item_idx, items_info, time):
    """
    Returns the next item to be presented based on activation.
    """

    # maps an item to its activation
    item_to_act = {}
    # calculate each item's activation
    for item in items:
        item_to_act[item] = calc_activation(item, items_info, time)

    # stores all items below the retention threshold
    endangered_items = []
    # for each item and its activation
    for k,v in item_to_act.items():
        # if the item's activation is BELOW the forgetting threshold
        if v < model_params["tau"]:
            # add it to the endangered list
            endangered_items.append(k)

    # if there ARE items BELOW the threshold
    if len(endangered_items) != 0:
        # find the endangered item with lowest activation
        next_item = min(endangered_items, key=item_to_act.get)
    # if ALL items are ABOVE the threshold
    # AND there ARE NEW items available
    elif next_new_item_idx < len(items):
        # select the next new item to be presented
        next_item = items[next_new_item_idx]
    # if NO items BELOW treshold
    # AND NO NEW items
    else:
        # find the item with the lowest activation
        next_item = min(item_to_act, key=item_to_act.get)

    return next_item, item_to_act[next_item], next_new_item_idx+1



def calc_activation(item, items_info, time):
    """
    Calculate the activation for a given item at a given timestamp.
    Takes into account all previous encounters of the item through the calculation of decay.

    Arguments:
    item -- the item whose activation should be calculated
    time -- the timestamp at which the activation should be calculated
    """

    # TODO: implement calculation algorithm
    return m



def calc_decay(item, items_info, time):
    """
    Calculate the activation decay of the given item at a given timestamp.
    Takes into account all previous encounters of the item.
    
    Argument:
    item -- the item, whose decay should be calculated
    time -- the timestamp at which the decay should be calculated
    """

    # TODO: implement 
    return d



def calc_recall_prob(activation):
    """
    Calculates the probability of recall given a specific activation.

    Arguments:
    activation -- the activation of the item, whose recall probability is being calculated.
    """

    Pr = np.divide(1, 1 + np.exp( np.divide(model_params["tau"] - activation, model_params["s"])))
    return Pr



def guess_item(recall_prob):
    """
    Guesses the word given a recall probability.

    Arguments:
    recall_prob -- the probablity that the given word can be recalled
    """

    return True if random.random() < recall_prob else False
