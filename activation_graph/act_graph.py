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
                "s":         0.255, # recall probability noise reduction factor
                "delta":     0.025} # factor to scale down intersession time


# list all items to be learned
items = ["noodles", "where", "many", "way", "market", "castle", "group", "restaurant", "amazing", "to visit", "each", "tree", "British", "adult", "a day", "open(from...to...)", "furniture", "a year", "open", "free time", "canal", "Chinese", "stall", "playing field", "fancy", "a week", "to enjoy", "best", "wonderful", "expensive", "to add", "boat", "to join in", "view", "canoeing", "flower", "area"] # end items list


# maps each item to its values
items_info = {}
for item in items:
    items_info[item] = Bunch(alpha_real=random.uniform(0.1, 0.5), alpha_model=model_params["alpha_d"], encounters=[], incorrect=0)


def print_item_info(item):
    print("Item:", item)
    print("Real Alpha:", items_info[item].alpha_real)
    print("Model Alpha:", items_info[item].alpha_model)
    print("Encounters(", len(items_info[item].encounters),"):")
    for i, enc in enumerate(items_info[item].encounters):
        print("Encounter", i, "time:", enc.time)
        print("Encounter", i, "activation:", enc.activation)
        print("Encounter", i, "recall prob:", calc_recall_prob(enc.activation))
        print("Encounter", i, "was guessed:", enc.was_guessed)
        print("Encounter", i, "alpha:", enc.alpha)
    print("Incorrect:", items_info[item].incorrect)


def graph_item_activation(item, items_info, learn_start):
    """
    Graphs the activation of the item throughout the learning process.
    item        -- the item, whose information is being graphed
    learn_start -- the time when the learning process started
    """

    plot_bounds = (0, 6000)
    first_seen = (items_info[item].encounters[0].time - learn_start).total_seconds()
    x = []
    y = []
    # for each second where the item was seen
    for i in range(int(first_seen), plot_bounds[1]):
        cur_act = calc_activation(item, items_info, (learn_start + datetime.timedelta(seconds=i)))
        x.append(i)
        y.append(cur_act)

    # plot the recall threshold
    plt.plot([plot_bounds[0], plot_bounds[1]],[0.8, 0.8], color='k', linestyle='--')

    # plot the actual data
    plt.plot(x, y, color='g')

    # Set plot information
    plt.title("Plot for \'" + item + "\'")
    plt.xlabel("Elapsed Session Time (seconds)")
    plt.ylabel("Item Activation")
    plt.axis([plot_bounds[0], plot_bounds[1],-1.5,1.5])
    # show the plot
    plt.show()
    return



def learn(items, items_info, sesh_count, sesh_length):
    """
    Simulates learning process by adding new encounters of words.
    Arguments:
    items       -- the items which need to be learned
    items_info  -- the information related to each item
    sesh_count  -- the number of sessions to be performed
    sesh_length -- the length of each session(in seconds)
    Returns:
    the datetime when the learning process started
    """

    # store the current time
    cur_time = datetime.datetime.now()
    learn_start = cur_time
    # store the index of the next NEW item which needs to be learned
    next_new_item_idx = 0

    # for each session
    for sesh_id in range(sesh_count):
        # set the session's starting time
        sesh_start = cur_time
        print("\n")
        print("Session", sesh_id, "start:", sesh_start)

        # while there is time in the session
        while cur_time <= (sesh_start + datetime.timedelta(seconds=sesh_length)):
            # get the next item to be presented
            item, item_act, next_new_item_idx = get_next_item(items, next_new_item_idx, items_info, cur_time)
            print("\nEncountered '", item, "' at", cur_time)
            print("Encounters:", len(items_info[item].encounters))
            print("Activation:", item_act)

            # calculate the item's recall probability
            item_rec = calc_recall_prob(item_act)
            print("Recall prob:", item_rec)
            # try to guess the item
            guessed = guess_item(item_rec)
            print("Guessed item:", guessed)

            # add the current encounter to the item's encounter list
            items_info[item].encounters.append(Bunch(time=cur_time, activation=item_act, was_guessed=guessed, alpha=items_info[item].alpha_model))

            # adjust values depending on outcome
            if guessed:
                if items_info[item].alpha_model > model_params["alpha_min"]:
                    items_info[item].alpha_model -= 0.08
            else:
                if items_info[item].alpha_model < model_params["alpha_max"]:
                    items_info[item].alpha_model += 0.08
                items_info[item].incorrect += 1

            # increment the current time to account for the length of the encounter
            cur_time += datetime.timedelta(seconds=random.randint(3, 15))

        # increment the current time to account for the intersession time
        scaled_intersesh_time = (24 * model_params["delta"]) * 3600
        cur_time += datetime.timedelta(seconds=scaled_intersesh_time)



    print("\nFinal results:")
    for item in items:
        print("Item:'", item, "'")
        print("Alpha:", items_info[item].alpha_model)
        print("Encounters:", len(items_info[item].encounters))
        print("Incorrect:", items_info[item].incorrect)

    print("Cur time:", cur_time)
    print("Learn start:", learn_start)
    return learn_start



def get_next_item(items, next_new_item_idx, items_info, time):
    """
    Returns the next item to be presented based on activation.
    """

    # store the index of the next new item
    next_new_item_idx_inc = next_new_item_idx

    # maps an item to its activation
    item_to_act = {}
    # calculate each SEEN items' activations
    for item in items[:next_new_item_idx_inc]:
        item_to_act[item] = calc_activation(item, items_info, time)

    # stores all items below the retention threshold
    endangered_items = []
    print("\nFinding next word:")
    # for each item and its activation
    for k,v in item_to_act.items():
        # calculate the item's probability of recall
        item_rec_prob = calc_recall_prob(v)
        # if the item's probability of recall is BELOW the forgetting threshold
        if v < model_params["tau"]:
            # add it to the endangered list
            endangered_items.append(k)

    # if there ARE items BELOW the threshold
    if len(endangered_items) != 0:
        # find the endangered item with lowest activation
        next_item = min(endangered_items, key=item_to_act.get)
        print("Item BELOW threshold!")
    # if ALL items are ABOVE the threshold
    # AND there ARE NEW items available
    elif next_new_item_idx_inc < len(items):
        # select the next new item to be presented
        next_item = items[next_new_item_idx_inc]
        # increment the index of the next new item
        next_new_item_idx_inc += 1
        print("Item is NEW word!")
    # if NO items BELOW treshold
    # AND NO NEW items
    else:
        # find the item with the lowest activation
        next_item = min(item_to_act, key=item_to_act.get)
        print("Item has LOWEST activation!")

    next_item_act = np.NINF if next_item not in item_to_act else item_to_act[next_item]

    return next_item, next_item_act, next_new_item_idx_inc



def calc_activation(item, items_info, cur_time):
    """
    Calculate the activation for a given item at a given timestamp.
    Takes into account all previous encounters of the item through the calculation of decay.

    Arguments:
    item       -- the item whose activation should be calculated
    items_info -- the information related to each item
    time       -- the timestamp at which the activation should be calculated
    """

    encounters = list(items_info[item].encounters)
    # stores the index of the previous encounter
    last_enc_idx = -1
    # for each encounter
    for enc_idx, enc_bunch in enumerate(encounters):
        # if the encounter was BEFORE the current time
        if enc_bunch.time < cur_time:
            # store the encounter's index
            last_enc_idx = enc_idx

    # if there are NO previous encounters
    if last_enc_idx == -1:
        m = np.NINF
    else:
        # stores the sum of time differences
        sum = 0.0
        # take only encounters which occurred before this one
        encounters = encounters[:(last_enc_idx+1)]
        # for each previous encounter
        for enc_idx, enc_bunch in enumerate(encounters):
            # calculate the time used in the activation formula
            future_time = cur_time + datetime.timedelta(seconds=15);
            # calculate the time difference with the previous encounter
            time_diff = future_time - enc_bunch.time
            # convert the difference into seconds
            time_diff = time_diff.total_seconds()

            # calculate the decay for the item at the current encounter
            decay = calc_decay(item, items_info, enc_idx)
            # SCALE the difference by the decay and ADD it to the sum
            sum += np.power(time_diff, -decay)

        # calculate the activation given the sum of scaled time differences
        m = np.log(sum)

    return m



def calc_decay(item, items_info, enc_idx):
    """
    Calculate the activation decay of the given item at a given encounter.
    
    Argument:
    item       -- the item, whose decay should be calculated
    items_info -- the information related to each item
    enc_idx    -- index of the encounter, at which the decay should be calculated
    """

    # stores the item's alpha value
    alpha = 0.0
    # get the item's activation at the encounter
    item_act = items_info[item].encounters[enc_idx].activation
    # if the activation is -infinity (the item hasn't been encountered before)
    if np.isneginf(item_act):
        # the alpha is the default value
        alpha = model_params["alpha_d"]
    else:
        # the alpha is the item's alpha
        alpha = items_info[item].alpha_real

    # calculate the decay
    d = model_params["c"] * np.exp(item_act) + alpha

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

    return True if random.uniform(0,1) < recall_prob else False
