# Tries to compare the result of calculating the activation using the CACHED values with the usual calculation
# Considers time spent, compares activations and final alphas


from bunch import Bunch
import datetime
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
items_info_uncached = {}
items_info_cached   = {}
for item in items:
    # Fill in the uncached info for the item
    items_info_uncached[item] = Bunch(alpha_real=random.uniform(model_params["alpha_min"], model_params["alpha_max"]), alpha_model=model_params["alpha_d"], encounters=[], incorrect=0)
    # Fill in the cached info for the item
    items_info_cached[item]   = Bunch(alpha_real=random.uniform(model_params["alpha_min"], model_params["alpha_max"]), alpha_model=model_params["alpha_d"], encounters=[], incorrect=0)



def test_act_call_count(item, items_info, enc_count, cached):
    """
    Prints how many times the activation function will be called given a number of encounters.
    NOTE: ALWAYS do learning before trying to count the calls
    """

    encounters    = items_info[item].encounters[:enc_count]
    last_enc_time = encounters[enc_count-1].time

    start = datetime.datetime.now()
    _, act_count = calc_activation(item, items_info[item].alpha_model, encounters, [], last_enc_time + datetime.timedelta(seconds=5))
    end   = datetime.datetime.now()

    print("\n")
    print("Item:", item)
    print("Encounters:", enc_count)
    print("The activation function was called", act_count, "times")
    print("Time taken:", (end - start).total_seconds())



def test_learn_duration(items, item_count, items_info, sesh_count, sesh_length, cached):
    """
    Prints how much time it takes for learning to finish given a number of items.
    """

    start = datetime.datetime.now()
    learn(items[:item_count], items_info, sesh_count, sesh_length)
    end   = datetime.datetime.now()

    enc_count = sum([len(items_info[item].encounters) for item in items[:item_count]])

    print("\n")
    print("Item count:", item_count)
    print("Total encounters:", enc_count)
    print("Duration:", (end - start).total_seconds())



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
        prev_encounters = [prev_enc for prev_enc in item_encounters if prev_enc.time < enc.time]
        print("Encounter", i, "time:",        enc.time)
        print("Encounter", i, "alpha:",       enc.alpha)
        enc_activation = calc_activation(item, enc.alpha, prev_encounters, [], enc.time)[0]
        print("Encounter", i, "activation:",  enc_activation)
        print("Encounter", i, "recall prob:", calc_recall_prob(enc_activation))
        print("Encounter", i, "was guessed:", enc.was_guessed)
    print("Incorrect:", items_info[item].incorrect)



def learn(items, items_info, sesh_count, sesh_length, cached):
    """
    Simulates the learning process by adding new encounters of words.
    Arguments:
    items       -- the items which need to be learned
    items_info  -- the information related to each item
    sesh_count  -- the number of sessions to be performed
    sesh_length -- the length of each session(in seconds)
    cached      -- whether to use the cached activations for each encounter
                   in the activation calculation OR to recalculate them again
    Returns:
    the datetime when the learning process started
    """

    # store the current time
    cur_time = datetime.datetime.now()
    # store the index of the next NEW item which needs to be learned
    # NOTE: if the new items are in a separate list, this does NOT need to be here
    next_new_item_idx = 0

    # for each session
    # NOTE: Only needed for simulating full learning process. Otherwise all learning takes place in a single session.
    for sesh_id in range(sesh_count):
        # set the session's starting time
        sesh_start = cur_time
        sesh_end   = sesh_start + datetime.timedelta(seconds=sesh_length)
        print("\nSession", sesh_id, "start:", sesh_start)

        # while there is time in the session
        while cur_time < sesh_end:
            # get the next item to be presented
            item, item_act_model, next_new_item_idx = get_next_item(items, items_info, cur_time, next_new_item_idx, cached)
            print("\nEncountered '", item, "' at", cur_time)

            # NOTE: Only needed to simulate user interaction
            # NOTE: NEVER use cached activations when calculating REAL activation
            # calculate the item's activation with its REAL alpha
            item_act_real = calc_activation(item, items_info[item].alpha_real, items_info[item].encounters, [], cur_time)[0]
            # calculate the 
            item_rec = calc_recall_prob(item_act_real)
            # try to guess the item
            guessed = guess_item(item_rec)

            # add the current encounter to the item's encounter list
            items_info[item].encounters.append(Bunch(time=cur_time, alpha=items_info[item].alpha_model, activation=item_act_model, was_guessed=guessed))

            # NOTE: The values for adjusting the alpha were selected through a bunch of testing and fitting
            # NOTE: in order for the results, produced by this simple simulation, to somewhat make sense.
            # adjust values depending on outcome
            if guessed:
                if items_info[item].alpha_model > model_params["alpha_min"]:
                    items_info[item].alpha_model -= 0.08
            else:
                if items_info[item].alpha_model < model_params["alpha_max"]:
                    items_info[item].alpha_model += 0.08
                items_info[item].incorrect += 1

            # increment the current time to account for the length of the encounter
            # NOTE: Only here to simulate the duration if interactions.
            cur_time += datetime.timedelta(seconds=random.randint(3, 10))

        # increment the current time to account for the intersession time
        # NOTE: Only here to simulate learning during multiple sessions.
        scaled_intersesh_time = (24 * model_params["delta"]) * 3600
        cur_time += datetime.timedelta(seconds=scaled_intersesh_time)



    print("\nFinal results:")
    for item in items:
        print("Item:'", item, "'")
        print("Alpha Real:", items_info[item].alpha_real)
        print("Alpha Model:", items_info[item].alpha_model)
        print("Encounters:", len(items_info[item].encounters))
        print("Incorrect:", items_info[item].incorrect)



def get_next_item(items, items_info, time, next_new_item_idx, cached):
    """
    Finds the next item to be presented based on their activation.
    Arguments:
    items             -- the items to be considered when choosing the next item
    items_info        -- the map, containing each item's information
    time              -- the time, at which the next item should be presented
    next_new_item_idx -- the index of the next NEW item from the list
    cached            -- whether to use the cached activations for each encounter
                         in the activation calculation OR to recalculate them again
    Returns:
    the next item to be presented, its activation and the index of the next NEW item
    """

    # extract all SEEN items
    # NOTE: If the SEEN and UNSEEN items were separate, this would NOT be needed
    seen_items = items[:next_new_item_idx]
    # maps an item to its activation
    item_to_act = {}
    # add 15 seconds to the time in order to catch items before they fall below the retrieval threshold
    future_time = time + datetime.timedelta(seconds=15)
    # recalculate each SEEN item's activation at future time with their updated alphas
    for item in seen_items:
        # Extract all encounters, which happened before future time
        prev_encounters  = [enc for enc in items_info[item].encounters if enc.time < future_time]
        # Store each previous encounter's activation
        prev_activations = [] if not cached else [enc.activation for enc in prev_encounters]
        item_to_act[item] = calc_activation(item, items_info[item].alpha_model, prev_encounters, prev_activations, future_time)[0]

    print("\nFinding next word!")

    # the default next item is the one with the lowest activation
    next_item = "" if len(seen_items) == 0 else min(seen_items, key=item_to_act.get)
    # stores the index of the next new item
    next_new_item_idx_inc = next_new_item_idx

    # if ALL items are NEW
    if next_item == "":
        # select the next new item to be presented
        next_item = items[next_new_item_idx]
        # increment the index of the next new item
        next_new_item_idx_inc += 1
    # if the lowest activation is ABOVE the retrieval threshold
    # AND
    # there ARE NEW items available
    elif item_to_act[next_item] > model_params["tau"] and next_new_item_idx < len(items):
        # select the next new item to be presented
        next_item = items[next_new_item_idx]
        # increment the index of the next new item
        next_new_item_idx_inc += 1

    # NOTE: in ALL other cases, the item with the lowest activation is selected
    # NOTE: (regardless of whether it is below the retrieval threshold)

    # Store the next item's activation based on whether it is a NEW item or NOT
    next_item_act = item_to_act[next_item] if next_item in item_to_act else calc_activation(next_item, items_info[next_item].alpha_model, [enc for enc in items_info[next_item].encounters if enc.time < time], [], future_time)[0]

    return next_item, next_item_act, next_new_item_idx_inc



def calc_activation(item, alpha, encounters, activations, cur_time, call_count=1):
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

    # NOTE: only here to count recursive function calls
    # stores the incremented call count when the function is called recursively
    call_count_inc = call_count

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
            # if the encounter's activation has NOT been calculated yet
            else:
                # calculate the activation of the item at the time of the encounter
                enc_act, call_count_inc = calc_activation(item, alpha, encounters[:enc_idx], activations, enc_bunch.time, call_count_inc)
                # NOTE: only here to count recursive function calls
                call_count_inc += 1
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

    return m, call_count_inc



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

    return 1 / (1 + np.exp(((model_params["tau"] - activation) / model_params["s"])))



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
