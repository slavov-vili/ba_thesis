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


def initialize_items_info(items):
    items_info = {}
    for item in items:
        items_info[item] = Bunch(alpha_real=random.gauss(0.3, 0.08),
                                 alpha_model=model_params["alpha_d"],
                                 encounters=[],
                                 incorrect=0)
    return items_info

# maps each item to its values
items_info = initialize_items_info(items)



def calc_avg_alpha_difference(items, items_info):
    sum_alpha_diff = 0
    for item in items:
        sum_alpha_diff += np.abs(items_info[item].alpha_real - items_info[item].alpha_model)
    return sum_alpha_diff / len(items)


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


def graph_item_info(item, items_info, learn_start):
    """
    Graphs the activation of the item throughout the learning process.
    Arguments:
    item        -- the item, whose activation is being graphed
    items_info  -- the map, containing each item's information
    learn_start -- the time when the learning process started
    Returns:
    nothing, shows the graph
    """

    # set the graph bounds
    plot_bounds_x_time  = (0, 6000)
    plot_bounds_y_act   = (-1.5, 1.5)
    plot_bounds_y_alpha = (model_params["alpha_min"], model_params["alpha_max"])

    # store the x and y values for activations and alphas
    x_act = []
    y_act = []
    y_alpha = []

    item_encounters = items_info[item].encounters
    # add the first encounter's information
    x_act.append(calc_time_diff(item_encounters[0].time, learn_start))
    y_act.append(item_encounters[0].activation)
    y_alpha.append(item_encounters[0].alpha)

    # for each two consecutive encounters
    for i in range(0, len(item_encounters)-1):
        prev_enc = item_encounters[i]
        next_enc = item_encounters[i+1]

        cur_time = prev_enc.time + datetime.timedelta(seconds=1)
        # for each second between the two encounters
        while cur_time < next_enc.time:
            # calculate the item's activation and add it to the graphing lists
            cur_act = calc_activation(item, prev_enc.alpha, [enc for enc in item_encounters if enc.time < cur_time], [], cur_time)
            # add the encounter's activation and alpha at its specific time
            x_act.append(calc_time_diff(cur_time, learn_start))
            y_act.append(cur_act)
            y_alpha.append(prev_enc.alpha)
            # increment the current time
            cur_time += datetime.timedelta(seconds=1)

        # add the next encounter's information
        x_act.append(calc_time_diff(next_enc.time, learn_start))
        y_act.append(next_enc.activation)
        y_alpha.append(next_enc.alpha)

    # create a subplot for activations
    plt.subplot(2,1,1)
    # plot the recall threshold
    plt.plot([plot_bounds_x_time[0], plot_bounds_x_time[1]], [model_params["tau"], model_params["tau"]], color='r', linestyle='--')
    # plot the item's activations
    plt.plot(x_act, y_act, color='g')
    # Set plot information
    plt.title("Plot for \'" + item + "\'")
    plt.ylabel("Item Activation")
    # set the plot's axes
    plt.axis([plot_bounds_x_time[0], plot_bounds_x_time[1], plot_bounds_y_act[0], plot_bounds_y_act[1]])

    # create a subplot for alphas
    plt.subplot(2,1,2)
    # plot the item's actual alpha
    plt.plot([plot_bounds_x_time[0], plot_bounds_x_time[1]], [items_info[item].alpha_real, items_info[item].alpha_real], color='k', linestyle='--')
    # plot the item's alphas
    plt.plot(x_act, y_alpha, color='b')
    # Set plot information
    plt.xlabel("Elapsed Session Time (seconds)")
    plt.ylabel("Item Alpha")
    # set the plot's axes
    plt.axis([plot_bounds_x_time[0], plot_bounds_x_time[1], plot_bounds_y_alpha[0], plot_bounds_y_alpha[1]])

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

            # add the current encounter to the item's encounter list
            items_info[item].encounters.append(Bunch(time=cur_time, alpha=items_info[item].alpha_model, activation=item_act, was_guessed=guessed))

            # adjust values depending on outcome
            if guessed:
                if items_info[item].alpha_model > model_params["alpha_min"]:
                    items_info[item].alpha_model -= 0.03
            else:
                if items_info[item].alpha_model < model_params["alpha_max"]:
                    items_info[item].alpha_model += 0.03
                items_info[item].incorrect += 1

            # increment the current time to account for the length of the encounter
            cur_time += datetime.timedelta(seconds=random.randint(3, 10))

        # increment the current time to account for the intersession time
        scaled_intersesh_time = (24 * model_params["delta"]) * 3600
        cur_time += datetime.timedelta(seconds=scaled_intersesh_time)



    print("\nFinal results:")
    for item in items:
        print("Item:'", item, "'")
        print("Alpha Real:", items_info[item].alpha_real)
        print("Alpha Model:", items_info[item].alpha_model)
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

    # extract all SEEN items
    seen_items = items[:next_new_item_idx]
    # maps an item to its activation
    item_to_act = {}
    # add 15 seconds to the time in order to catch items before the fall below the retrieval threshold
    future_time = time + datetime.timedelta(seconds=15)
    # recalculate each SEEN item's activation at future time with their updated alphas
    for item in seen_items:
        item_to_act[item] = calc_activation(item, items_info[item].alpha_model, [enc for enc in items_info[item].encounters if enc.time < future_time], [], future_time)

    print("\nFinding next word:")

    # the default next item is the one with the lowest activation
    next_item = "" if len(seen_items) == 0 else min(seen_items, key=item_to_act.get)
    # stores the index of the next new item
    next_new_item_idx_inc = next_new_item_idx

    # if ALL items are NEW
    if next_item == "":
        # select the next new item to be presented
        next_item = items[next_new_item_idx_inc]
        # increment the index of the next new item
        next_new_item_idx_inc += 1
        print("Item is NEW word!")
    # if the lowest activation is BELOW the retrieval threshold
    elif item_to_act[next_item] < model_params["tau"]:
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
        print("Item has LOWEST activation!")

    # store the next item's activation based on whether it is a NEW item or NOT
    next_item_act = calc_activation(next_item, items_info[next_item].alpha_model, [enc for enc in items_info[next_item].encounters if enc.time < time], [], future_time) if next_item not in item_to_act else item_to_act[next_item]

    return next_item, next_item_act, next_new_item_idx_inc



def calc_activation(item, alpha, encounters, activations, cur_time):
    """
    Calculates the activation for a given item at a given timestamp.
    Takes into account all previous encounters of the item through the calculation of decay.
    Arguments:
    item        -- the item whose activation should be calculated
    alpha       -- the alpha value of the item which should be used in the calculation
    encounters  -- the list of all of the item's encounters
    activations -- the list of activations corresponding to each encounter (used for caching activation values)
    cur_time    -- the timestamp at which the activation should be calculated
    Returns:
    the activation of the item at the given timestamp
    """

    # if there are NO previous encounters
    if len(encounters) == 0:
        m = np.NINF
    # ASSUMING that the encounters are sorted according to their timestamps
    # if the last encounter happens later than the time of calculation
    elif encounters[len(encounters)-1].time > cur_time:
        raise ValueError("Encounters must happen BEFORE the time of activation calculation!")
    else:
        # stores the sum of time differences
        sum = 0.0
        # for each encounter
        for enc_idx, enc_bunch in enumerate(encounters):
            # stores the encounter's activation
            enc_act   = 0.0
            # if the encounter's activation has ALREADY been calculated
            if enc_idx < len(activations):
                enc_act = activations[enc_idx]
            # if the encounter's activation has NOT been calculated
            else:
                # calculate the activation of the item at the time of the encounter
                enc_act = calc_activation(item, alpha, [enc for enc in encounters if enc.time < enc_bunch.time], activations, enc_bunch.time)
                # add the encounter's activation to the list
                activations.append(enc_act)

            # calculate the time difference between the current time and the previous encounter
            # AND convert it to seconds
            time_diff = calc_time_diff(cur_time, enc_bunch.time)
            # calculate the item's decay at the encounter
            enc_decay = calc_decay(enc_act, alpha)

            # SCALE the difference by the decay and ADD it to the sum
            sum += np.power(time_diff, -enc_decay)

        # calculate the activation given the sum of scaled time differences
        m = np.log(sum)

    return m



def calc_decay(activation, alpha):
    """
    Calculate the activation decay of an item given its activation and alpha at the time of encounter.
    Arguments:
    activation -- the activation of the item
    alpha      -- the alpha of the item
    Returns:
    the decay value of the item
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
    Calculates an item's probability of recall given its activation.
    Arguments:
    activation -- the activation of the item
    Returns:
    the item's probability of recall based on activation
    """

    # calculate the probability of recall
    Pr = 1 / (1 + np.exp(((model_params["tau"] - activation) / model_params["s"])))
    return Pr



def calc_time_diff(cur_time, start_time):
    """
    Calculates the difference between a starting time and the current time.
    The time difference is represented in total seconds.
    """
    return (cur_time - start_time).total_seconds()



def guess_item(recall_prob):
    """
    Guesses the word given a recall probability.
    Arguments:
    recall_prob -- the probablity that the given word can be recalled
    Returns:
    True if the word was guessed, False otherwise
    """

    return True if random.uniform(0,1) < recall_prob else False
