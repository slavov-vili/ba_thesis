from bunch import Bunch
import datetime
import numpy             as np



model_params = {"alpha_default":             0.3,   # default alpha for all items
                "alpha_min":                 0.1,   # minimum possible alpha
                "alpha_max":                 0.5,   # maximum possible alpha
                "decay_scale":               0.21,  # decay scaling factor
                "threshold":                -0.8,   # activation threshold
                "rec_prob_noise_reduction":  0.255, # recall probability noise reduction factor
                "intersesh_time_scale":      0.025} # factor to scale down intersession time


# A list     of all items, which the user has NOT yet encountered, BUT NEED to be learned
items_new  = [item_form]
# A dictionary of all items, which the user HAS encountered, BUT still NEED to be learned
items_seen = {item_form: item_info}

# An ITEM'S FORM is represented simply by a string
item_form_sample = ""
# An ITEM'S INFORMATION is represented by the following values:
item_info_sample = Bunch(cur_alpha=0.0,
                         encounters=[])
# An ENCOUNTER  is represented by the following information:
encounter_sample = Bunch(time=datetime.datetime,
                         alpha=0.0,
                         result="")





# Execution DEPENDS on values of arguments
#
# B   = 4 = [get current time, assign session start, +, assign session end] // BASIC instructions, always executed by the FUNCTION
# SNS = len(known items seen this session)
# AEC                                                                       // Average ENCOUNTER count for SEEN ITEMS
# NNS = len(new items seen this session)
# A   = SNS + NNS                                                           // Instructions, executed when APPENDING encounters
# SL                                                                        // The session LENGTH
# AEL                                                                       // AVERAGE encounter LENGTH (same unit as session length)
# NI                                                                        // Instructions, when finding NEXT ITEM
# ENC                                                                       // Instructions, when ENCOUNTERING an item
# RP                                                                        // Instructions, when calculating RECALL PROBABILITY
# NA                                                                        // Instructions, when calculating NEW ALPHA
# W   = 9 + NI + A + RP + NA = [get current time, assign current time, if,  // BASIC instructions, ALWAYS executed in the WHILE loop
#                               assign new item, ENC, store encounter result,
#                               store current alpha, create new Bunch,
#                               assign new alpha] + NI + A + RP + NA
#
# INSTR_COUNT = B + ((SL / AEL) + 1) * ()
#
def learn(items_seen, items_new, session_length):
    """
    Goes through a single learning session.
    Arguments:
    items_seen     -- a dictionary of items, which the user HAS    encountered before, but still hasn't learned
    items_new      -- a list       of items, which the user HASN'T encountered before, but still needs to learn
    session_length -- the length of the study session (in seconds)
    Returns:
    updated lists of new and seen items
    """

    # store the session's START time
    session_start = datetime.datetime.now()
    # store the session's END   time
    session_end   = session_start + datetime.timedelta(seconds=session_length)

    # while there is still time in the session
    while True:
        cur_time = datetime.datetime.now()
        if cur_time >= session_end:
            break

        # get the next item to be presented
        cur_item, cur_item_act, items_new = get_next_item(items_seen, items_new, cur_time)

        # Encounter the item and store the encounter's result
        cur_enc_result = encounter_item(cur_item)
        # Store the current item's alpha
        cur_item_alpha = items_seen[cur_item].cur_alpha
        # Add the current encounter to the item's encounter list
        items_seen[cur_item].encounters.append(Bunch(time=cur_time,
                                                     alpha=cur_item_alpha,
                                                     result=cur_enc_result))

        # Adjust the alpha value depending on the result of the current encounter
        items_seen[cur_item].cur_alpha = calc_new_alpha(cur_item_alpha, calc_recall_prob(cur_item_act), cur_enc_result)





def get_next_item(items_seen, items_new, cur_time):
    """
    Finds the next item to be presented based on their activation.
    Arguments:
    items_seen -- a dictionary of items, which the user HAS    encountered before, but still hasn't learned
    items_new  -- a list       of items, which the user HASN'T encountered before, but still needs to learn
    cur_time   -- the time at which the next item will be presented
    Returns:
    the next item to be presented, its activation and the possible updated list of new items
    """

    # Maps an item to its activation
    item_to_act = {}
    # Add 15 seconds to the time in order to catch items before they fall below the retrieval threshold
    future_time = cur_time + datetime.timedelta(seconds=15)

    # Recalculate each SEEN item's activation at future time with their updated alphas
    for item in items_seen:
        item_to_act[item] = calc_activation(item, items_seen[item].cur_alpha, [enc for enc in items_seen[item].encounters if enc.time < future_time], [], future_time)

    # The default next item is the one with the lowest activation
    next_item = "" if len(seen_items) == 0 else min(items_seen, key=item_to_act.get)

    # If ALL items are NEW
    if next_item == "":
        # Select the next NEW item to be presented
        next_item = items_new[0]
    # If the lowest activation is ABOVE the retrieval threshold
    # AND
    # There ARE NEW items available
    elif item_to_act[next_item] > model_params["threshold"] and len(items_new) != 0:
        # select the next new item to be presented
        next_item = items_new[0]

    # NOTE: In ALL other cases, the item with the lowest activation is selected
    # NOTE: (regardless of whether it is below the retrieval threshold)

    # Stores the next item's activation
    next_item_act = 0.0

    # If the next item is NOT NEW
    if next_item in item_to_act:
        next_item_act = item_to_act[next_item]

    # If the next item IS NEW
    else:
        # Remove the selected item from the new item list
        items_new.remove(next_item)
        # Initialize the item in the 'seen' list
        items_seen[next_item] = Bunch(cur_alpha=model_params["alpha_default"], encounters=[])
        # Calculate the item's activation
        next_item_act = calc_activation(next_item, items_seen[next_item].cur_alpha, [], [], future_time)

    return next_item, next_item_act





# Execution DEPENDS on values of arguments
#
# NOTE: WITHOUT caching, the function is recursively called a total of 2 ^ N times
# NOTE: WITH    caching, the function is recursively called a total of 1 + N times
# NOTE: triang_num = factorial, but using ADDITION
#
# N   = len(encounters)
# T   = 5                                                        // Instructions, when calculating the TIME DIFFERENCE
# D   = 4 | 5                                                    // Instructions, when calculating the DECAY
# A   = 1                                                        // Instructions, when APPENDING an encounter's activation to CACHE
# S   = triang_num(N) = [create new list,                        // Instructions, when SLICING off previous encounters
#                        add all prev encounters]
# 
# B   = 2 = [declare activation, return activation]              // BASIC instructions, ALWAYS executed by the function
# BC  = 4 = B + 2 = B + [if, assign activation]                  // BASIC instructions, ALWAYS executed when the BASE CASE is reached
# E_1 = 5 = [if, elif, declare sum, log, assign activation]      // BASIC instructions, ALWAYS executed in the FIRST ELSE statement
# F   = 18|19 = 9 + 5 + 4|5 = [assign enc_idx, assign enc_bunch, // BASIC instructions, ALWAYS executed in the FOR loop
#                              assign enc_time, declare enc_act,
#                              assign time_diff, assign decay,
#                              pow, +, assign sum] + T + D
# I_2 = 2 = [if, assign enc_act]                                 // BASIC instructions, ALWAYS executed in the SECOND IF statement
# E_2 = 2 = [if, S, assign enc_act, A]                           // BASIC instructions, ALWAYS executed in the SECOND IF statement
# 
# |======================================================================================================================|
# |        CONDITIONS        |              INSTR COUNT             |    CUR INSTR LIST    |  TOTAL INSTR COUNT          |
# |======================================================================================================================|
# | N != 0 &                 |       B + 3                          | [if, if,             |          5                  |
# | last_enc.time > cur_time |                                      |  raise error]        |                             |
# |--------------------------|--------------------------------------|----------------------|-----------------------------|
# | else                     | B + E_1 + (N*F) + BC +               | []                   | 2^(N+1) + (19|20)*N +       |
# |                          | (((2^N) - (1+N)) * I_2) +            |                      | triang_num(N) + 9           |
# |                          | (N * (I_2 + assign enc_act)) + S + A |                      |                             |
# |======================================================================================================================|
def calc_activation(item, cur_alpha, encounters, activations, cur_time):
    """
    Calculates the activation for a given item at a given timestamp.
    Takes into account all previous encounters of the item through the calculation of decay.
    Arguments:
    item        -- the item whose activation should be calculated
    cur_alpha   -- the alpha value of the item which should be used in the calculation
    encounters  -- the list of all of the item's encounters
    activations -- the list of activations corresponding to each encounter (used for caching activation values)
    cur_time    -- the timestamp at which the activation should be calculated
    Returns:
    the activation of the item at the given timestamp
    """

    # Stores the item's activation
    activation = 0.0

    # If there are NO previous encounters
    if len(encounters) == 0:
        activation = np.NINF
    # ASSUMING that the encounters are sorted according to their timestamps
    # If the last encounter happens later than the time of calculation
    elif encounters[len(encounters)-1].time > cur_time:
        raise ValueError("Encounters must happen BEFORE the time of activation calculation!")
    else:
        # Stores the sum of time differences
        sum = 0.0

        # For each encounter
        for enc_idx, enc_bunch in enumerate(encounters):
            # Store the encounter's time
            enc_time = enc_bunch.time
            # Stores the item's activation at that encounter
            enc_act = 0.0

            # If the encounter's activation has ALREADY been calculated
            if enc_idx < len(activations):
                enc_act = activations[enc_idx]

            # If the encounter's activation has NOT been calculated yet
            else:
                # Calculate the activation of the item at the time of the encounter
                enc_act = calc_activation(item, cur_alpha, encounters[:enc_idx], activations, enc_time)
                # Add the current encounter's activation to the list
                activations.append(enc_act)

            # Calculate the time difference between the current time and the previous encounter
            # AND convert it to seconds
            time_diff = calc_time_diff(cur_time, enc_time)
            # calculate the item's decay at the encounter
            enc_decay = calc_decay(enc_act, alpha)

            # SCALE the difference by the decay and ADD it to the sum
            sum = sum + np.power(time_diff, -enc_decay)

        # calculate the activation given the sum of scaled time differences
        activation = np.log(sum)

    return activation





# Execution DEPENDS on values of arguments
#
# B = 2 = [declare decay, return decay]    // BASIC instructions, ALWAYS executed by the function
#
# |=========================================================================================|
# |       CONDITIONS        |    INSTR COUNT   |    CUR INSTR LIST    |   TOTAL INSTR COUNT |
# |=========================================================================================|
# | is_neg_inf              |       B + 2      | [if, assign decay]   |          4          |
# |-------------------------|------------------|----------------------|---------------------|
# | else                    |       B + 4      | [if, *, +,           |          6          |
# |                         |                  |  assign decay]       |                     |
# |=========================================================================================|
def calc_decay(activation, cur_alpha):
    """
    Calculate the activation decay of an item given its activation at the time of an encounter and its alpha.
    Arguments:
    activation -- the activation of the item at the given encounter
    cur_alpha  -- the current alpha of the item
    Returns:
    the decay value of the item
    """

    decay = 0.0

    # if the activation is -infinity (the item hasn't been encountered before)
    if np.isneginf(activation):
        # the decay is the default alpha value
        decay = model_params["alpha_default"]
    else:
        # calculate the decay
        decay = model_params["decay_scale"] * np.exp(activation) + cur_alpha

    return decay





# Executes in CONSTANT time
#
# B = 5 = [-, /, exp, +, /]    // BASIC instructions, ALWAYS executed by the function
#
def calc_recall_prob(activation):
    """
    Calculates an item's probability of recall given its activation.
    Arguments:
    activation -- the activation of the item
    Returns:
    the item's probability of recall based on activation
    """

    return 1 / (1 + np.exp(((model_params["threshold"] - activation) / model_params["rec_prob_noise_reduction"])))





# Execution DEPENDS on values of arguments
#
# B = 3 = [assign time_diff, calc_total_seconds, return time_diff]    // BASIC instructions, ALWAYS executed by the function
#
# |=========================================================================================|
# |       CONDITIONS        |    INSTR COUNT   |    CUR INSTR LIST    |   TOTAL INSTR COUNT |
# |=========================================================================================|
# | time_a > time_b         |       B + 2      | [if, -]              |          5          |
# |-------------------------|------------------|----------------------|---------------------|
# | else                    |       B + 2      | [if, -]              |          5          |
# |=========================================================================================|
# NOTE: This calculation gets more complicated once sessions come into play
def calc_time_diff(time_a, time_b):
    """
    Calculates the difference between two timestamp.
    The time difference is represented in total seconds.
    Arguments:
    time_a -- the first  timestamp to be considered
    time_b -- the second timestamp to be considered
    Returns:
    the time difference in seconds
    """

    time_diff = (time_a - time_b) if time_a > time_b else (time_b - time_a)

    return (time_diff).total_seconds()





# Executes in CONSTANT time
#
# B = 1 = [return result]    // BASIC instructions, ALWAYS executed by the function
#
# NOTE: This calculation depends on what the item encounters consist of
def encounter_item(item):
    """
    Presents the item to the user and records the outcome of the encounter
    Arguments:
    item -- the item to be presented
    Returns:
    'guessed', 'not_guessed' or 'skipped' depending on what the outcome of the encounter was
    """

    return "guessed / not_guessed / skipped"





# Execution DEPENDS on values of arguments
#
# B = 2 = [assign new_alpha, return new_alpha]    // BASIC instructions, ALWAYS executed by the function
#
# |=========================================================================================|
# |       CONDITIONS        |    INSTR COUNT   |    CUR INSTR LIST    |   TOTAL INSTR COUNT |
# |=========================================================================================|
# | "skipped"               |      B + 2       | [if, elif]           |          4          |
# |=========================================================================================|
# | "guessed" &             |      B + 6       | [if, if, -, /, -,    |          8          |
# | > alpha_min             |                  |  assign new_alpha]   |                     |
# |-------------------------|------------------|----------------------|---------------------|
# | "guessed" &             |      B + 2       | [if, if]             |          4          |
# | < alpha_min             |                  |                      |                     |
# |=========================================================================================|
# | "not_guessed" &         |      B + 6       | [if, elif, if, /, +, |          8          |
# | < alpha_max             |                  |  assign new_alpha]   |                     |
# |-------------------------|------------------|----------------------|---------------------|
# | "not_guessed" &         |      B + 3       | [if, elif, if]       |          5          |
# | > alpha_max             |                  |                      |                     |
# |=========================================================================================|
# NOTE: This calculation may get more complicated depending on how the alpha is adjusted
def calc_new_alpha(old_alpha, item_recall_prob, last_enc_result):
    """
    Calculates the new alpha value, based on the result of the last encounter with the item and the item's recall_probability
    Arguments:
    old_alpha        -- the item's old alpha value
    last_enc_result  -- the result of the user's last encounter with the item
    item_recall_prob -- the item's recall_probability during the last encounter
    Returns:
    the item's new alpha value
    """

    new_alpha = old_alpha

    # If the last encounter was SUCCESSFUL
    if last_enc_result == "guessed":
        if old_alpha > model_params["alpha_min"]:
            new_alpha = new_alpha - (1 - item_recall_prob) / 40

    # If the last encounter was NOT SUCCESSFUL
    elif last_enc_result == "not_guessed":
        if old_alpha < model_params["alpha_max"]:
            new_alpha = new_alpha + item_recall_prob / 40

    # NOTE: If the last encounter was SKIPPED, the alpha stays the same

    return new_alpha
