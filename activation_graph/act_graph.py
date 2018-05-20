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


def print_item_info(item, items_info):
    """
    Prints out all the information for a specific item.
    Arguments:
    item       -- the item, whose information will be printed
    items_info -- the map, containing each item's information
    Returns:
    nothing, prints to stdout
    """
    print("Item:", item)
    print("Real Alpha:",  items_info[item].alpha_real)
    print("Model Alpha:", items_info[item].alpha_model)
    item_encounters = items_info[item].encounters
    print("Encounters(", len(item_encounters),"):")
    for i, enc in enumerate(item_encounters):
        print("Encounter", i, "time:",        enc.time)
        print("Encounter", i, "alpha:",       enc.alpha)
        print("Encounter", i, "activation:",  enc.activation)
        print("Encounter", i, "recall prob:", calc_recall_prob(enc.activation))
        print("Encounter", i, "was guessed:", enc.was_guessed)
    print("Incorrect:", items_info[item].incorrect)


def graph_item_activation(item, items_info, learn_start):
    """
    Graphs the activation of the item throughout the learning process.
    Arguments:
    item        -- the item, whose activation is being graphed
    items_info  -- the map, containing each item's information
    learn_start -- the time when the learning process started
    Returns:
    nothing, shows the graph
    """

    plot_bounds_x = (0, 6000)
    plot_bounds_y = (-1.5, 1.5)
    x = []
    y = []
    item_encounters = items_info[item].encounters
    # add the first encounter's information
    x.append((item_encounters[0].time - learn_start).total_seconds())
    y.append(item_encounters[0].activation)

    # for each two consecutive encounters
    for i in range(0, len(item_encounters)-1):
        prev_enc = item_encounters[i]
        next_enc = item_encounters[i+1]
        time_between_encounters = (next_enc.time - prev_enc.time).total_seconds()

        secs = 1
        # for each second between the two encounters
        while secs < time_between_encounters:
            cur_time = prev_enc.time + datetime.timedelta(seconds=secs)
            # calculate the item's activation and add it to the graphing lists
            cur_act = calc_activation(item, prev_enc.alpha, item_encounters, [], cur_time)
            x.append((cur_time - learn_start).total_seconds())
            y.append(cur_act)
            # increment the second counter
            secs += 1

        # add the next encounter's information
        x.append((next_enc.time - learn_start).total_seconds())
        y.append(next_enc.activation)

    # plot the recall threshold
    plt.plot([plot_bounds_x[0], plot_bounds_x[1]], [model_params["tau"], model_params["tau"]], color='k', linestyle='--')
    # plot the actual data
    plt.plot(x, y, color='g')

    # Set plot information
    plt.title("Plot for \'" + item + "\'")
    plt.xlabel("Elapsed Session Time (seconds)")
    plt.ylabel("Item Activation")
    plt.axis([plot_bounds_x[0], plot_bounds_x[1], plot_bounds_y[0], plot_bounds_y[1]])
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
    learn_start = datetime.datetime.now()
    cur_time    = learn_start
    # store the index of the next NEW item which needs to be learned
    next_new_item_idx = 0

    # for each session
    for sesh_id in range(sesh_count):
        # set the session's starting time
        sesh_start = cur_time
        sesh_end   = sesh_start + datetime.timedelta(seconds=sesh_length)
        print("\n")
        print("Session", sesh_id, "start:", sesh_start)

        # while there is time in the session
        while cur_time < sesh_end:
            # get the next item to be presented
            item, item_act, next_new_item_idx = get_next_item(items, items_info, cur_time, next_new_item_idx)
            print("\nEncountered '", item, "' at", cur_time)
            print("Encounters:", len(items_info[item].encounters))
            print("Activation:", item_act)

            # calculate the item's recall probability with its REAL alpha
            item_rec = calc_recall_prob(calc_activation(item, items_info[item].alpha_real, items_info[item].encounters, [], cur_time))
            print("Recall prob:", item_rec)
            # try to guess the item
            guessed = guess_item(item_rec)
            print("Guessed item:", guessed)

            # adjust values depending on outcome
            if guessed:
                if items_info[item].alpha_model > model_params["alpha_min"]:
                    items_info[item].alpha_model -= item_rec / 10
            else:
                if items_info[item].alpha_model < model_params["alpha_max"]:
                    items_info[item].alpha_model += item_rec / 10
                items_info[item].incorrect += 1

            # add the current encounter to the item's encounter list
            items_info[item].encounters.append(Bunch(time=cur_time, alpha=items_info[item].alpha_model, activation=item_act, was_guessed=guessed))

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
    return learn_start



def get_next_item(items, items_info, time, next_new_item_idx):
    """
    Finds the next item to be presented based on their activation.
    Arguments:
    items             -- the items to be considered when choosing the next item
    items_info        -- the map, containing each item's information
    time              -- the time, at which the next item should be presented
    next_new_item_idx -- the index of the next NEW item from the list
    Returns:
    the next item to be presented, its activation and the index of the next NEW item
    """

    # store the index of the next new item
    next_new_item_idx_inc = next_new_item_idx

    # maps an item to its activation
    item_to_act = {}
    # TODO: calculate activation for 15 seconds in future
    # recalculate each SEEN item's activation at current time with their updated alphas
    for item in items[:next_new_item_idx_inc]:
        item_to_act[item] = calc_activation(item, items_info[item].alpha_model, items_info[item].encounters, [], time)

    # stores all items below the retention threshold
    endangered_items = []
    print("\nFinding next word:")
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



# TODO: remove time + 15 seconds
def calc_activation(item, alpha, encounters, activations, cur_time):
    """
    Calculate the activation for a given item at a given timestamp.
    Takes into account all previous encounters of the item through the calculation of decay.

    Arguments:
    item        -- the item whose activation should be calculated
    alpha       -- the alpha value of the item which should be used in the calculation
    encounters  -- the list of all of the item's encounters
    activations -- the list of old activations corresponding to each encounter
    cur_time    -- the timestamp at which the activation should be calculated
    """

    # stores the index of the last encounter BEFORE the current timestamp
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
        prev_encounters = encounters[:(last_enc_idx+1)]
        # for each previous encounter
        for enc_idx, enc_bunch in enumerate(prev_encounters):
            # store the encounter's information
            enc_time  = enc_bunch.time
            enc_act   = 0.0
            # if the encounter's activation has ALREADY been calculated
            if enc_idx < len(activations):
                enc_act = activations[enc_idx]
            # if the encounter's activations has NOT been calculated
            else:
                # calculate the activation of the item at the time of the encounter
                enc_act = calc_activation(item, alpha, encounters, activations, enc_time)
                # add the encounter's activation to the list
                activations.append(enc_act)

            # calculate the time used in the activation formula
            future_time = cur_time + datetime.timedelta(seconds=15);
            # calculate the time difference with the previous encounter
            time_diff = future_time - enc_time
            # convert the difference into seconds
            time_diff = time_diff.total_seconds()
            # calculate the item's decay at the encounter
            enc_decay = calc_decay(enc_act, alpha)

            # SCALE the difference by the decay and ADD it to the sum
            sum += np.power(time_diff, -enc_decay)

        # calculate the activation given the sum of scaled time differences
        m = np.log(sum)

    return m



def calc_decay(activation, alpha):
    """
    Calculate the activation decay of the given item at a given encounter.
    
    Argument:
    item       -- the item, whose decay should be calculated
    items_info -- the information related to each item
    enc_idx    -- index of the encounter, at which the decay should be calculated
    """

    # if the activation is -infinity (the item hasn't been encountered before)
    if np.isneginf(activation):
        # the decay is the default alpha value
        d = model_params["alpha_d"]
    else:
        # calculate the decay
        d = model_params["c"] * np.exp(activation) + alpha

    return d



def calc_recall_prob(activation):
    """
    Calculates an item's probability of recall given a timestamp and its real alpha.

    Arguments:
    item       -- the item whose activation should be calculated
    alpha      -- the alpha value of the item which should be used in the calculation
    encounters -- the list of all of the item's encounters
    cur_time   -- the timestamp at which the activation should be calculated
    """

    # calculate the probability of recall
    Pr = 1 / (1 + np.exp(((model_params["tau"] - activation) / model_params["s"])))
    return Pr



def guess_item(recall_prob):
    """
    Guesses the word given a recall probability.

    Arguments:
    recall_prob -- the probablity that the given word can be recalled
    """

    return True if random.uniform(0,1) < recall_prob else False
