# Tries to compare the result of calculating the activation using the CACHED values with the usual calculation
# Considers time spent, compares activations and final alphas


from bunch import Bunch
import datetime
import numpy    as np
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

def reset_items_info(items_info):
    for item in items_info:
        items_info[item].alpha_model = model_params["alpha_d"]
        items_info[item].encounters  = []
        items_info[item].incorrect   = 0

# maps each item to its values
items_info = initialize_items_info(items)



def initialize_avg_items_info(items):
    avg_items_info = {}
    for item in items:
        avg_items_info[item] = Bunch(avg_enc_count   = 0.0,
                                     avg_perc_incorr = 0.0,
                                     avg_alpha       = 0.0)
    return avg_items_info


# TODO: maybe implement encounter-specific information to track performance more closely
def test_learning(items, items_info, sesh_count, sesh_length, learn_count, cached, immediate_alpha, alpha_adjustment, cache_update):
    """
    Gets average values for learning both when using the cached and the uncached activations for each encounter
    Prints a comparison between the two averages.

    learn_count  -- how many times should learning sessions be conducted when getting the average values
    """

    # Store the average values for learning sessions
    averages = Bunch(avg_duration    = 0.0,
                     avg_enc_count   = 0.0,
                     avg_alpha_err   = 0.0,
                     avg_items_info  = initialize_avg_items_info(items))

    # Conduct learning sessions
    for i in range(learn_count):
        # Reset the session-specific item information
        reset_items_info(items_info)

        # Conduct a learning session
        start = datetime.datetime.now()
        learn(items, items_info, sesh_count, sesh_length, cached, immediate_alpha, alpha_adjust_value)
        end   = datetime.datetime.now()

        # Add the LEARNING session's DURATION to the averages
        learn_duration = (end - start).total_seconds()
        averages.avg_duration += learn_duration
        # Add the LEARNING session's ENCOUNTER COUNT to the averages
        learn_enc_count = sum([len(items_info[item].encounters) for item in items])
        averages.avg_enc_count += learn_enc_count

        # Add each item
        for item in items:
            item_enc_count = len(items_info[item].encounters)
            # Add the item's ENCOUNTER COUNT to the averages
            averages.avg_items_info[item].avg_enc_count   += item_enc_count
            # Add the item's INCORRECT PERCENTAGE to the averages
            averages.avg_items_info[item].avg_perc_incorr += (items_info[item].incorrect / item_enc_count) * 100
            # Add the item's ALPHA to the ITEM-SPECIFIC averages
            averages.avg_items_info[item].avg_alpha       += items_info[item].alpha_model


    # Calculate the averages
    averages.avg_duration  /= learn_count
    averages.avg_enc_count /= learn_count
    for item in items:
        averages.avg_items_info[item].avg_enc_count   /= learn_count
        averages.avg_items_info[item].avg_perc_incorr /= learn_count
        averages.avg_items_info[item].avg_alpha       /= learn_count
        # Add the item's calculated average alpha error to the session's average
        averages.avg_alpha_err += np.abs(items_info[item].alpha_real - averages.avg_items_info[item].avg_alpha)
    averages.avg_alpha_err /= len(items)


    return averages



def test_act_call_count(item, items_info, enc_count, cached):
    """
    Prints how many times the activation function will be called given a number of encounters.
    NOTE: ALWAYS do learning before trying to count the calls in order to get a list of encounters
    """

    encounters    = items_info[item].encounters[:enc_count]
    activations   = [] if not cached else [enc.activation for enc in encounters]
    last_enc_time = encounters[enc_count-1].time

    start = datetime.datetime.now()
    _, act_count = calc_activation(item, items_info[item].alpha_model, encounters, activations, last_enc_time + datetime.timedelta(seconds=5))
    end   = datetime.datetime.now()

    print("\n")
    print("Item:", item)
    print("Encounters:", enc_count)
    print("The activation function was called", act_count, "times")
    print("Time taken:", (end - start).total_seconds())


def calc_avg_alpha_difference(items, items_info):
    sum_alpha_diff = 0
    for item in items:
        sum_alpha_diff += np.abs(items_info[item].alpha_real - items_info[item].alpha_model)
    return sum_alpha_diff / len(items)



def print_item_info(item, items_info, cached):
    """
    Prints out all the information for a specific item.
    Arguments:
    item       -- the item, whose information will be printed
    items_info -- the map, containing each item's information
    Returns:
    nothing, prints to stdout
    """
    print("Item:", item)
    print("Real Alpha: ",  items_info[item].alpha_real)
    print("Model Alpha:", items_info[item].alpha_model)
    item_encounters = items_info[item].encounters
    print("Encounters: ", len(item_encounters))
    print("Incorrect:  ", items_info[item].incorrect)
    for i, enc in enumerate(item_encounters):
        prev_encounters  = [prev_enc for prev_enc in item_encounters if prev_enc.time < enc.time]
        prev_activations  = [] if not cached else [enc.activation for enc in prev_encounters]
        print("Encounter", i, "time:",        enc.time)
        print("Encounter", i, "alpha:",       enc.alpha)
        print("Encounter", i, "activation:",  enc.activation)
        print("Encounter", i, "recall prob (real): ",
               calc_recall_prob(calc_activation(item, items_info[item].alpha_real, prev_encounters, prev_activations, enc.time)[0]))
        print("Encounter", i, "recall prob (model):", calc_recall_prob(enc.activation))
        print("Encounter", i, "was guessed:", enc.was_guessed)



def learn(items, items_info, sesh_count, sesh_length, cached, immediate_alpha_adjustment, alpha_adjust_value, cache_update):
    """
    Simulates the learning process by adding new encounters of words.

    Arguments:
    items       -- the items which need to be learned
    items_info  -- the information related to each item
    sesh_count  -- the number of sessions to be performed
    sesh_length -- the length of each session (in seconds)
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

    # for each STUDY session
    # NOTE: Only needed for simulating full learning process. Otherwise all learning takes place in a single session.
    for sesh_id in range(sesh_count):
        # Set the session's information
        sesh_start = cur_time
        sesh_end   = sesh_start + datetime.timedelta(seconds=sesh_length)
        sesh_enc_counts       = {}
        sesh_incorrect_counts = {}
        sesh_avg_recall_probs = {}
        for item in items:
            sesh_enc_counts[item]       = 0
            sesh_incorrect_counts[item] = 0
            sesh_avg_recall_probs[item] = 0.0
        # print("\nSession", sesh_id, "start:", sesh_start)

        # while there is time in the session
        while cur_time < sesh_end:
            # get the next item to be presented
            item, item_act_model, next_new_item_idx = get_next_item(items, items_info, cur_time, next_new_item_idx, cached)
            # print("Encountered '", item, "' at", cur_time)

            # NOTE: Only needed to simulate user interaction
            # calculate the item's activation with its REAL alpha
            # NOTE: NEVER use cached activations when calculating the item's REAL activation
            item_act_real = calc_activation(item, items_info[item].alpha_real, items_info[item].encounters, [], cur_time)[0]
            # calculate the item's recall probability, based on the its real activation
            item_rec_real = calc_recall_prob(item_act_real)
            # try to guess the item
            guessed = guess_item(item_rec_real)

            # add the current encounter to the item's encounter list
            items_info[item].encounters.append(Bunch(time=cur_time,
                                                     alpha=items_info[item].alpha_model,
                                                     activation=item_act_model,
                                                     was_guessed=guessed))

            # NOTE: The values for adjusting the alpha were selected through a bunch of testing and fitting
            # NOTE: in order for the results, produced by this simple simulation, to somewhat make sense.
            item_rec_model = calc_recall_prob(item_act_model)
            # If the alpha should be adjusted immediately after an encounter
            if immediate_alpha_adjustment:
                # adjust the alpha value depending on the outcome
                if guessed:
                    if items_info[item].alpha_model > model_params["alpha_min"]:
                        items_info[item].alpha_model -= (1 - item_rec_model) * (2 * alpha_adjust_value)
                else:
                    if items_info[item].alpha_model < model_params["alpha_max"]:
                        items_info[item].alpha_model += item_rec_model * (2 * alpha_adjust_value)
                    # Increment the ITEM's incorrect counter
                    items_info[item].incorrect  += 1
                    # Increment the SESSION's incorrect counter
                    sesh_incorrect_counts[item] += 1

                # If the cached activations should be used
                if cached:
                    # Extract all encounters, which happened before future time
                    item_encounters = [enc for enc in items_info[item].encounters if enc.time < cur_time]
                    for 
            elif not guessed:
                # Add the recall probability to the session average
                sesh_avg_recall_probs[item] += item_rec_model
                # Increment the ITEM's incorrect counter
                items_info[item].incorrect  += 1
                # Increment the SESSION's incorrect counter
                sesh_incorrect_counts[item] += 1

            # increment the session's encounter count
            sesh_enc_counts[item] += 1

            # increment the current time to account for the length of the encounter
            # NOTE: Only here to simulate the duration if interactions.
            cur_time += datetime.timedelta(seconds=random.randint(3, 10))

        # If the alphas should be updated AFTER the end of the SESSION
        if not immediate_alpha_adjustment:
            # For each item
            for item in items:
                # If the item WAS seen in this session
                if sesh_enc_counts[item] != 0:
                    sesh_avg_recall_probs[item] /= sesh_enc_counts[item]
                    # If the overall outcome of the session is a SUCCESS
                    if sesh_incorrect_counts[item] < (sesh_enc_counts[item] / 2):
                        # Adjust the alpha to INCREASE the decay
                        if items_info[item].alpha_model < model_params["alpha_max"]:
                            items_info[item].alpha_model -= (1 - sesh_avg_recall_probs[item]) * (2 * alpha_adjust_value)
                    # If the model predicts a LOWER activation
                    else:
                        # Adjust the alpha to DECREASE the decay
                        if items_info[item].alpha_model > model_params["alpha_min"]:
                            items_info[item].alpha_model += sesh_avg_recall_probs[item] * (2 * alpha_adjust_value)


        # increment the current time to account for the intersession time
        # NOTE: Only here to simulate learning during multiple sessions.
        scaled_intersesh_time = (24 * model_params["delta"]) * 3600
        cur_time += datetime.timedelta(seconds=scaled_intersesh_time)



    # print("\nFinal results:")
    # for item in items:
        # print("Item:'", item, "'")
        # print("Alpha Real:", items_info[item].alpha_real)
        # print("Alpha Model:", items_info[item].alpha_model)
        # print("Encounters:", len(items_info[item].encounters))
        # print("Incorrect:", items_info[item].incorrect)



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
    future_time = calc_future_time(time)
    # recalculate each SEEN item's activation at future time with their updated alphas
    for item in seen_items:
        # Extract all encounters, which happened before future time
        prev_encounters   = [enc for enc in items_info[item].encounters if enc.time < future_time]
        # Store each previous encounter's activation if cached values should be used
        prev_activations  = [] if not cached else [enc.activation for enc in prev_encounters]
        # Map each item to its activation
        item_to_act[item] = calc_activation(item, items_info[item].alpha_model, prev_encounters, prev_activations, future_time)[0]

    # print("\nFinding next word!")

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
    # NOTE: IF the activation needs to be RECALCULATED, NO CACHED values for activations are passed, SINCE the item would be NEW
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



def calc_future_time(time):
    """
    Adds a given amount to the time
    """
    return time + datetime.timedelta(seconds=15)



def guess_item(recall_prob):
    """
    Guesses the word given a recall probability.
    Arguments:
    recall_prob -- the probablity that the given word can be recalled
    Returns:
    True if the word was guessed, False otherwise
    """

    return True if random.random() < recall_prob else False





# def main():
    # learn_sesh_counts       = [50] # [5, 10, 15, 20, 25, 50, 100]
    # study_sesh_counts       = [4] #[2, 4, 8]
    # study_sesh_lengths      = [1800, 3600]
    # alpha_adjust_values     = [0.03] #[0.01, 0.03, 0.05, 0.08, 0.1]
    # immediate_alpha_adjusts = [True] #, False]

    # # For all possible numbers of learning sessions
    # for learn_sesh_count in learn_sesh_counts:
        # # For all possible numbers of study sessions
        # for study_sesh_count in study_sesh_counts:
            # # For all possible session lengths
            # for study_sesh_length in study_sesh_lengths:
                # # For all possible values of alpha adjusting
                # for alpha_adjust_value in alpha_adjust_values:
                    # # For all possible ways of adjusting the alpha
                    # for immediate_alpha_adjust in immediate_alpha_adjusts:
                        # reset_items_info(items_info)

                        # print("\n\n\n\n")
                        # print("Testing learning:")
                        # print("Learn sessions         =", learn_sesh_count)
                        # print("Study sessions         =", study_sesh_count)
                        # print("Study session length   =", study_sesh_length)
                        # print("Immediate alpha adjust =", immediate_alpha_adjust)
                        # print("Alpha adjust value     =", alpha_adjust_value)

                        # test_learning(items, items_info, study_sesh_count, study_sesh_length, learn_sesh_count, immediate_alpha_adjust, alpha_adjust_value)


    # print("\n")
    # print("FULL Differences:")
    # print("Average Duration (UNCACHED)        = ", averages_uncached.avg_duration, "SECONDS")
    # print("Average Duration (CACHED)          = ", averages_cached.avg_duration,   "SECONDS")
    # print("Average Encounter Count (UNCACHED) = ", averages_uncached.avg_enc_count, "ENCOUNTERS")
    # print("Average Encounter Count (CACHED)   = ", averages_cached.avg_enc_count,   "ENCOUNTERS")
    # print("Average Alpha Error (UNCACHED)     = ", averages_uncached.avg_alpha_err)
    # print("Average Alpha Error (CACHED)       = ", averages_cached.avg_alpha_err)
    # for item in items:
        # avg_item_info_uncached = averages_uncached.avg_items_info[item]
        # avg_item_info_cached   = averages_cached.avg_items_info[item]
        # print("\nItem:", item)
        # print("Average Encounter Count   (UNCACHED) = ", avg_item_info_uncached.avg_enc_count,   "ENCOUNTERS")
        # print("Average Encounter Count   (CACHED)   = ", avg_item_info_cached.avg_enc_count,     "ENCOUNTERS")
        # print("Average Percentage Incorr (UNCACHED) = ", avg_item_info_uncached.avg_perc_incorr, "PERCENT")
        # print("Average Percentage Incorr (CACHED)   = ", avg_item_info_cached.avg_perc_incorr,   "PERCENT")
        # print("Alpha         (Real)                 = ", items_info[item].alpha_real)
        # print("Average Alpha (Model)     (UNCACHED) = ", avg_item_info_uncached.avg_alpha)
        # print("Average Alpha (Model)     (CACHED)   = ", avg_item_info_cached.avg_alpha)

# main()
