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
                "delta":     0.025, # factor to scale down intersession time
                "F":         1,     # scaling factor for the reaction time
                "f":         0.3}     # base reaction time



def initialize_items_info(items):
    items_info = {}
    for item in items:
        items_info[item] = Bunch(alpha_real  = random.gauss(0.3, 0.08),
                                 alpha_model = model_params["alpha_d"],
                                 encounters  = [],
                                 incorrect   = 0)
    return items_info

def reset_items_info(items_info):
    for item in items_info:
        items_info[item].alpha_model = model_params["alpha_d"]
        items_info[item].encounters  = []
        items_info[item].incorrect   = 0

def initialize_avg_items_info(items):
    avg_items_info = {}
    for item in items:
        avg_items_info[item] = Bunch(avg_enc_count   = 0.0,
                                     avg_perc_incorr = 0.0,
                                     avg_alpha       = 0.0)
    return avg_items_info

def initialize_enc_items(enc_count):
    """
    Initializes a list, which stores lists of items.
    Each list represents all items which have been seen at the given encounter index throughout multiple LEARNING sessions.
    """

    enc_items = []
    for enc_idx in range(enc_count):
        enc_items.append([])
    return enc_items

def initialize_item_to_encs(items):
    item_to_encs = {}
    for item in items:
        item_to_encs[item] = []
    return item_to_encs


# list all items to be learned
items = ["noodles", "where", "many", "way", "market", "castle", "group", "restaurant", "amazing", "to visit", "each", "tree", "British", "adult", "a day", "open(from...to...)", "furniture", "a year", "open", "free time", "canal", "Chinese", "stall", "playing field", "fancy", "a week", "to enjoy", "best", "wonderful", "expensive", "to add", "boat", "to join in", "view", "canoeing", "flower", "area"] # end items list
# maps each item to its values
items_info = initialize_items_info(items)





# [ Testing area ]

def test_learning(items, items_info, sesh_count, sesh_length, learn_count, cached, immediate_alpha, alpha_adjust_value, cache_update, inter_sesh_time):
    """
    Gets average values for learning both when using the cached and the uncached activations for each encounter
    Prints a comparison between the two averages.

    learn_count  -- how many times should learning sessions be conducted when getting the average values
    """

    # Store the average values for learning sessions
    averages = Bunch(avg_duration    = 0.0,
                     avg_enc_count   = 0.0,
                     avg_alpha_err   = 0.0,
                     avg_alpha_bias  = 0.0,
                     avg_items_info  = initialize_avg_items_info(items))

    # Conduct learning sessions
    for i in range(learn_count):
        # Reset the session-specific item information
        reset_items_info(items_info)

        # Conduct a learning session
        start = datetime.datetime.now()
        learn(items, items_info, sesh_count, sesh_length, cached, immediate_alpha, alpha_adjust_value, cache_update, inter_sesh_time, False)
        end   = datetime.datetime.now()

        # Add the LEARNING session's DURATION to the averages
        learn_duration = (end - start).total_seconds()
        averages.avg_duration += learn_duration
        # Add the LEARNING session's ENCOUNTER COUNT to the averages
        learn_enc_count = sum([len(items_info[item].encounters) for item in items])
        averages.avg_enc_count  += learn_enc_count
        # Add the LEARNING session's ALPHA ERROR to the averages
        averages.avg_alpha_err  += calc_avg_alpha_error(items, items_info)
        # Add the LEARNING session's ALPHA BIAS to the averages
        averages.avg_alpha_bias += calc_avg_alpha_bias(items, items_info)
        # Add each item
        for item in items:
            item_enc_count = len(items_info[item].encounters)
            # Add the item's ENCOUNTER COUNT to the averages
            averages.avg_items_info[item].avg_enc_count   += item_enc_count
            # Add the item's INCORRECT PERCENTAGE to the averages
            averages.avg_items_info[item].avg_perc_incorr += (items_info[item].incorrect / item_enc_count) * 100 if item_enc_count != 0 else 0
            # Add the item's ALPHA to the ITEM-SPECIFIC averages
            averages.avg_items_info[item].avg_alpha       += items_info[item].alpha_model

    # Calculate the averages
    averages.avg_duration   /= learn_count
    averages.avg_enc_count  /= learn_count
    averages.avg_alpha_err  /= learn_count
    averages.avg_alpha_bias /= learn_count
    for item in items:
        averages.avg_items_info[item].avg_enc_count   /= learn_count
        averages.avg_items_info[item].avg_perc_incorr /= learn_count
        averages.avg_items_info[item].avg_alpha       /= learn_count


    print("\n")
    print("Average Duration        = ", averages.avg_duration, "SECONDS")
    print("Average Encounter Count = ", averages.avg_enc_count, "ENCOUNTERS")
    print("Average Alpha Error     = ", averages.avg_alpha_err)
    print("Average Alpha Bias      = ", averages.avg_alpha_bias)
    for item in averages.avg_items_info:
        avg_item_info = averages.avg_items_info[item]
        print("\nItem:", item)
        print("Average Encounter Count   = ", avg_item_info.avg_enc_count,   "ENCOUNTERS")
        print("Average Percentage Incorr = ", avg_item_info.avg_perc_incorr, "PERCENT")
        print("Alpha         (Real)      = ", items_info[item].alpha_real)
        print("Average Alpha (Model)     = ", avg_item_info.avg_alpha)

def test_learn_duration(items, item_count, items_info, sesh_count, sesh_length, cached, immediate_alpha_adjustment, alpha_adjust_value, cache_update):
    """
    Prints how much time it takes for learning to finish given a number of items.
    """
    # Reset the session-specific item information
    reset_items_info(items_info)

    start = datetime.datetime.now()
    learn(items[:item_count], items_info, sesh_count, sesh_length, cached, immediate_alpha_adjustment, alpha_adjust_value, cache_update)
    end   = datetime.datetime.now()

    enc_count = sum([len(items_info[item].encounters) for item in items[:item_count]])

    print("\n")
    print("Item count:", item_count)
    print("Total encounters:", enc_count)
    print("Duration:", (end - start).total_seconds())

def test_act_call_count(item, items_info, enc_count):
    """
    Prints how many times the activation function will be called given a number of encounters.
    NOTE: ALWAYS do learning before trying to count the calls
    """

    encounters    = items_info[item].encounters[:enc_count]

    start = datetime.datetime.now()
    _, act_count = calc_activation(item, items_info[item].alpha_model, encounters, [], start)
    end   = datetime.datetime.now()

    print("\n")
    print("Item:", item)
    print("Encounters:", enc_count)
    print("The activation function was called", act_count, "times")
    print("Time taken:", (end - start).total_seconds())

def test_encounter_history(items, items_info, sesh_count, sesh_length, inter_sesh_time):
    """
    Gets the encounter-specific results of learning sessions when using:
        - the real alphas    and the original  implementation of the algorithm
        - the model's alphas and the original  implementation of the algorithm
        - the model's alphas and the optimized implementation of the algorithm
    """

    learn_sessions = []

    # Reset the session-specific item information
    reset_items_info(items_info)
    # Do a learning session with the REAL ALPHAS, without USING the CACHED values
    learn(items, items_info, sesh_count, sesh_length, False, True, 0.02, "", 24, True, initialize_item_to_encs(items))
    # Store the session's encounter OUTCOMES
    encounter_outcomes = get_session_enc_outcomes(items, items_info)
    # Add the learning session's information to the list
    learn_sessions.append(Bunch(real_alphas  = True,
                                cache_used   = False,
                                cache_update = "-",
                                history      = get_session_encounters(items_info)))

    # Reset the session-specific item information
    reset_items_info(items_info)
    # Do a learning session with the MODEL's ALPHAS, WITHOUT using the CACHED values (using the previously stored encounter outcomes)
    learn(items, items_info, sesh_count, sesh_length, False, True, 0.02, "", 24, False, encounter_outcomes)
    # Add the learning session's information to the list
    learn_sessions.append(Bunch(real_alphas  = False,
                                cache_used   = False,
                                cache_update = "-",
                                history      = get_session_encounters(items_info)))

    # Reset the session-specific item information
    reset_items_info(items_info)
    # Do a learning session with the MODEL's ALPHAS, USING the CACHED values (using the previously stored encounter outcomes)
    learn(items, items_info, sesh_count, sesh_length, True, True, 0.02, "post-session", 24, False, encounter_outcomes)
    # Add the learning session's information to the list
    learn_sessions.append(Bunch(real_alphas  = False,
                                cache_used   = True,
                                cache_update = "post-session",
                                history      = get_session_encounters(items_info)))

    # Use a history size which exists in all recorded session histories
    # history_size = min([len(sesh.history) for sesh in learn_sessions])
    # For each recorded session
    for i, sesh in enumerate(learn_sessions):
        print("Test Settings:")
        print("Real alphas used:", sesh.real_alphas)
        print("Cache used:      ", sesh.cache_used)
        print("Cache updated:   ", sesh.cache_update)
        # TODO: print bias
        print("Avg enc idx offset from next test:", calc_session_avg_enc_offset(sesh.history,      learn_sessions[(i+1)%3].history))
        print("Avg enc idx offset bias from next test:", calc_session_avg_enc_offset_bias(sesh.history, learn_sessions[(i+1)%3].history))

    item_to_enc_indices_real      = get_item_enc_indices(learn_sessions[0].history)
    item_to_enc_indices_model     = get_item_enc_indices(learn_sessions[1].history)
    item_to_enc_indices_optimized = get_item_enc_indices(learn_sessions[2].history)
    print("\n")
    print("Item History Comparison:")
    # For each item
    for item in items:
        # Print the item's encounter indices and the offset from the real values
        print("Item:", item)
        print("Encounter indices (real):     ", item_to_enc_indices_real[item])
        print("Average index offset from next test:", calc_item_avg_enc_offset(item,  item_to_enc_indices_real, item_to_enc_indices_model))
        print("Index offset bias    from next test:", calc_item_enc_offset_bias(item, item_to_enc_indices_real, item_to_enc_indices_model))
        print("Encounter indices (model):    ", item_to_enc_indices_model[item])
        print("Average index offset from next test:", calc_item_avg_enc_offset(item,  item_to_enc_indices_model, item_to_enc_indices_optimized))
        print("Index offset bias    from next test:", calc_item_enc_offset_bias(item, item_to_enc_indices_model, item_to_enc_indices_optimized))
        print("Encounter indices (optimized):", item_to_enc_indices_optimized[item])
        print("Average index offset from next test:", calc_item_avg_enc_offset(item,  item_to_enc_indices_optimized, item_to_enc_indices_real))
        print("Index offset bias    from next test:", calc_item_enc_offset_bias(item, item_to_enc_indices_optimized, item_to_enc_indices_real))

    print("\n")
    print("Encounter History Comparison:")
    biggest_history = max(len(sesh.history) for sesh in learn_sessions)
    # For each encounter
    for i in range(biggest_history):
        print("Encounter", i, ":")
        # Store the item, encountered at the current encounter in the respective history
        item_real      = learn_sessions[0].history[i] if i < len(learn_sessions[0].history) else "-"
        item_model     = learn_sessions[1].history[i] if i < len(learn_sessions[1].history) else "-"
        item_optimized = learn_sessions[2].history[i] if i < len(learn_sessions[2].history) else "-"
        # Print the respective items
        print("\titem_real_alphas         = ", item_real)
        print("\titem_model_alphas        = ", item_model)
        print("\titem_optimized_algorithm = ", item_optimized)



def get_session_enc_outcomes(items, items_info):
    # Maps an item to a list of the outcomes of its encounters
    item_to_encs = initialize_item_to_encs(items)
    # For each item
    for item in items:
        # For each encounter of that item
        for enc in items_info[item].encounters:
            # Add the encounter's outcome to the map
            item_to_encs[item].append(enc.was_guessed)
    return item_to_encs

def get_session_encounters(items_info):
    """
    Extracts the full encounter history of a given session

    Arguments:
    items_info -- the information for each item from that session

    Returns:
    the sorted list of encounters (tuples of encounter time and the item which got presented)
    """
    # Store each encounter and the item which got presented in a list of tuples
    encounters = []
    for item in items_info:
        for enc in items_info[item].encounters:
            encounters.append((enc.time, item))
    # Return the sorted list of encounters
    return [tup[1] for tup in sorted(encounters)]

def get_item_enc_indices(session_history):
    """
    Converts a session history to a map of item to all encounter indices, where that item occurs.
    """
    item_to_enc_indices = {}
    for item in session_history:
        item_to_enc_indices[item] = []
    for idx, item in enumerate(session_history):
        item_to_enc_indices[item].append(idx)
    return item_to_enc_indices





# [ Metadata calculation]

def calc_avg_alpha_error(items, items_info):
    sum_alpha_err = 0
    for item in items:
        sum_alpha_err += np.abs(items_info[item].alpha_real - items_info[item].alpha_model)
    return sum_alpha_err / len(items)

def calc_avg_alpha_bias(items, items_info):
    sum_alpha_bias = 0
    for item in items:
        sum_alpha_bias += items_info[item].alpha_real - items_info[item].alpha_model
    return sum_alpha_bias / len(items)

def calc_session_avg_enc_offset(hist_1, hist_2):
    """
    Calculates the average encounter index offset given two session histories.
    A history is a list of items and represent which item was seen at which encounter.

    NOTE: the function assumes, that both histories contain the same items, just in possibly different orders
    """


    # Transform the histories into maps of items to the lists of encounter indices, where they (the items) was seen
    item_to_indices_1 = get_item_enc_indices(hist_1)
    item_to_indices_2 = get_item_enc_indices(hist_2)

    # Stores the complete encounter index differences of all items
    total_avg_diff = 0

    # For each item
    for item in item_to_indices_1:
        # Add the average encounter difference for the item to the overall difference
        total_avg_diff += calc_item_avg_enc_offset(item, item_to_indices_1, item_to_indices_2)

    return total_avg_diff / len(item_to_indices_1)

def calc_item_avg_enc_offset(item, item_to_indices_1, item_to_indices_2):
    """
    Calculates the average encounter offset for the given item.

    NOTE: the function assumes, that both histories contain the same items, just in possibly different orders
    """
    avg_idx_diff = 0
    # Store the encounter indices of the item from the respective map
    item_encs_1 = item_to_indices_1[item]
    item_encs_2 = item_to_indices_2[item]
    # Get the length of the smaller encounter index list
    most_encs = max(len(item_encs_1), len(item_encs_2))
    # Add all index differences to the average
    for i in range(most_encs):
        encs_1_idx = item_encs_1[i] if i < len(item_encs_1) else 0
        encs_2_idx = item_encs_2[i] if i < len(item_encs_2) else 0
        avg_idx_diff += np.abs(encs_1_idx - encs_2_idx)
    return avg_idx_diff / most_encs

def calc_session_avg_enc_offset_bias(hist_1, hist_2):
    """
    Calculates the average encounter index offset bias given two session histories.
    A history is a list of items and represent which item was seen at which encounter.

    NOTE: the function assumes, that both histories contain the same items, just in possibly different orders
    """


    # Transform the histories into maps of items to the lists of encounter indices, where they (the items) was seen
    item_to_indices_1 = get_item_enc_indices(hist_1)
    item_to_indices_2 = get_item_enc_indices(hist_2)

    # Stores the complete encounter index differences of all items
    avg_idx_diff_bias = 0

    # For each item
    for item in item_to_indices_1:
        # Add the average encounter difference for the item to the overall difference
        avg_idx_diff_bias += calc_item_enc_offset_bias(item, item_to_indices_1, item_to_indices_2)

    return avg_idx_diff_bias / len(item_to_indices_1)

def calc_item_enc_offset_bias(item, item_to_indices_1, item_to_indices_2):
    """
    Calculates the encounter index offset bias for the given item.

    NOTE: the function assumes, that both histories contain the same items, just in possibly different orders
    """
    idx_bias = 0
    # Store the encounter indices of the item from the respective map
    item_encs_1 = item_to_indices_1[item]
    item_encs_2 = item_to_indices_2[item]
    # Get the length of the smaller encounter index list
    most_encs = max(len(item_encs_1), len(item_encs_2))
    # Add all index differences to the bias
    for i in range(most_encs):
        encs_1_idx = item_encs_1[i] if i < len(item_encs_1) else 0
        encs_2_idx = item_encs_2[i] if i < len(item_encs_2) else 0
        idx_bias += encs_1_idx - encs_2_idx
    return idx_bias





# [ Printing ]

def print_final_results(items_info, full=True):
    relevant_items = 0
    for item in items_info:
        if full:
            relevant_items.append(item)
        elif len(items_info[item].encounters) != 0:
            relevant_items.append(item)

    print("\nFinal results:")
    print("Average alpha error:", calc_avg_alpha_error(relevant_items, items_info))
    print("Average alpha bias: ", calc_avg_alpha_bias(relevant_items,  items_info))
    for item in relevant_items:
        print("Item:     '", item, "'")
        print("Alpha Real: ", items_info[item].alpha_real)
        print("Alpha Model:", items_info[item].alpha_model)
        print("Encounters: ", len(items_info[item].encounters))
        print("Incorrect:  ", items_info[item].incorrect)

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
    print("Real  Alpha:", items_info[item].alpha_real)
    print("Model Alpha:", items_info[item].alpha_model)
    print("Alpha Error:", items_info[item].alpha_real - items_info[item].alpha_model)
    item_encounters = items_info[item].encounters
    print("Encounters: ", len(item_encounters))
    print("Incorrect:  ", items_info[item].incorrect)
    for i, enc in enumerate(item_encounters):
        print("Encounter", i, "time:",       enc.time)
        print("Encounter", i, "alpha:",      enc.alpha)
        print("Encounter", i, "activation:", enc.activation)
        print("Encounter", i, "recall prob (real): ",
                calc_recall_prob(calc_activation(item, items_info[item].alpha_real, item_encounters[:i], [], enc.time)[0]))
        print("Encounter", i, "recall prob (model):", calc_recall_prob(enc.activation))
        print("Encounter", i, "was guessed:", enc.was_guessed)





# [ Implementation area ]


def learn(items, items_info, sesh_count, sesh_length, cached, immediate_alpha_adjustment, alpha_adjust_value, cache_update, inter_sesh_time, use_real_alphas, encounter_outcomes):
    """
    Simulates the learning process by adding new encounters of words.

    Arguments:
    items                      -- the items which need to be learned
    items_info                 -- the information related to each item
    sesh_count                 -- the number of sessions to be performed
    sesh_length                -- the length of each session (in seconds)
    cached                     -- whether to use the cached activations for each encounter
                                  in the activation calculation OR to recalculate them again
    immediate_alpha_adjustment -- whether the alpha should be adjusted immediately after each encounter
    alpha_adjust_value         -- how much should the alpha be adjusted
    cache_update               -- when should the cache be updated
    use_real_alphas            -- whether the real alphas of items should be used when getting the next item
    encounter_outcomes         -- a list of predetermined outcomes for all encounters

    Returns:
    the datetime when the learning process started
    """

    # If the real alphas of items should be used
    if use_real_alphas:
        # Set each item's alpha to be its real alpha
        for item in items:
            items_info[item].alpha_model = items_info[item].alpha_real

    # store the current time
    cur_time = datetime.datetime.now()
    # store the index of the next NEW item which needs to be learned
    # NOTE: if the new items are in a separate list, this does NOT need to be here
    next_new_item_idx = 0
    # store the index of the current encounter
    cur_enc_idx = 0

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
            item, next_new_item_idx = get_next_item(items, items_info, cur_time, next_new_item_idx, cached)
            # print("Encountered '", item, "' at", cur_time)

            # Extract all encounters, which happened before future time
            prev_encounters   = [enc for enc in items_info[item].encounters if enc.time < cur_time]
            # Store each previous encounter's activation if cached values should be used
            prev_activations  = [] if not cached else [enc.activation for enc in prev_encounters]
            # Calculate the item's activation with the MODEL's alpha
            item_act_model = calc_activation(item, items_info[item].alpha_model, prev_encounters, prev_activations, cur_time)[0]
            item_rec_model = calc_recall_prob(item_act_model)

            # NOTE: Only needed to simulate user interaction
            # calculate the item's activation with its REAL alpha
            # NOTE: NEVER use cached activations when calculating the item's REAL activation
            item_act_real = calc_activation(item, items_info[item].alpha_real, prev_encounters, [], cur_time)[0]
            # calculate the item's recall probability, based on the its real activation
            item_rec_real = calc_recall_prob(item_act_real)
            # Store the index of the current encounter
            cur_enc_idx = len(prev_encounters)
            # try to guess the item
            guessed = guess_item(item_rec_real) if cur_enc_idx >= len(encounter_outcomes[item]) else encounter_outcomes[item][cur_enc_idx]

            # add the current encounter to the item's encounter list
            items_info[item].encounters.append(Bunch(time        = cur_time,
                                                     alpha       = items_info[item].alpha_model,
                                                     activation  = item_act_model,
                                                     was_guessed = guessed))

            # NOTE: The values for adjusting the alpha were selected through a bunch of testing and fitting
            # NOTE: in order for the results, produced by this simple simulation, to somewhat make sense.
            # If the alpha should be adjusted immediately after an encounter
            if immediate_alpha_adjustment and not use_real_alphas:
                # adjust the alpha value depending on the outcome
                if guessed:
                    if items_info[item].alpha_model > model_params["alpha_min"]:
                        items_info[item].alpha_model -= (1 - item_rec_model) * (2 * alpha_adjust_value)
                else:
                    if items_info[item].alpha_model < model_params["alpha_max"]:
                        items_info[item].alpha_model += item_rec_model * (2 * alpha_adjust_value)
                    # Increment the ITEM's incorrect counter
                    items_info[item].incorrect  += 1

                # If the cached activations should be updated IMMEDIATELY after the alpha is changed
                if cache_update == "immediately":
                    # Store the item's encounters
                    item_encounters = items_info[item].encounters
                    # For each encounter
                    for i, enc in enumerate(item_encounters):
                        # Get all encounters, which happened before this one
                        prev_encounters  = item_encounters[:i]
                        # Get each encounter's activation
                        prev_activations = [prev_enc.activation for prev_enc in prev_encounters]
                        # recalculate the encounter's activation with the updated alpha
                        enc.activation   = calc_activation(item, items_info[item].alpha_model, prev_encounters, prev_activations, enc.time)[0]
            else:
                # Add the recall probability to the session average
                sesh_avg_recall_probs[item] += item_rec_model
                if not guessed:
                    # Increment the ITEM's incorrect counter
                    items_info[item].incorrect  += 1
                    # Increment the SESSION's incorrect counter
                    sesh_incorrect_counts[item] += 1

            # increment the session's encounter count
            sesh_enc_counts[item] += 1

            # NOTE: Only here to simulate the duration if interactions.
            # Encounter duration = time it takes to start typing + average time to type a word
            enc_duration = datetime.timedelta(seconds = calc_reaction_time(item_act_model)) + datetime.timedelta(seconds = 3.0)
            # increment the current time to account for the length of the encounter
            cur_time    += enc_duration

        # If the alphas should be updated AFTER the end of the SESSION
        if not immediate_alpha_adjustment and not use_real_alphas:
            # For each item
            for item in items:
                # If the item WAS seen in this session
                if sesh_enc_counts[item] != 0:
                    sesh_avg_recall_probs[item] /= sesh_enc_counts[item]

                    # TODO: consider better way to evaluate if session was success
                    # If the overall outcome of the session is a SUCCESS
                    if sesh_incorrect_counts[item] < (sesh_enc_counts[item] / 4):
                        # Adjust the alpha to INCREASE the decay
                        if items_info[item].alpha_model > model_params["alpha_min"]:
                            items_info[item].alpha_model -= (1 - sesh_avg_recall_probs[item]) * (2 * alpha_adjust_value)
                    # If the item was NOT seen
                    else:
                        # Adjust the alpha to DECREASE the decay
                        if items_info[item].alpha_model < model_params["alpha_max"]:
                            items_info[item].alpha_model += sesh_avg_recall_probs[item] * (2 * alpha_adjust_value)

        # If the cached activations should be updated AFTER the end of the SESSION
        if cache_update == "post-session":
            # For each item
            for item in items:
                # If the item WAS seen in this session
                if sesh_enc_counts[item] != 0:
                    # Store the item's encounters
                    item_encounters = items_info[item].encounters
                    # For each encounter
                    for i, enc in enumerate(item_encounters):
                        # NOTE: Since encounters are ordered chronologically,
                        #       the previous encounter list always contains updated values
                        # Get all previous encounters
                        prev_encounters  = item_encounters[:i]
                        # Get each previous encounter's activation
                        prev_activations = [prev_enc.activation for prev_enc in prev_encounters]
                        # recalculate the encounter's activation with the updated alpha
                        enc.activation   = calc_activation(item, items_info[item].alpha_model, prev_encounters, prev_activations, enc.time)[0]


        # increment the current time to account for the intersession time
        # NOTE: Only here to simulate learning during multiple sessions.
        scaled_intersesh_time = (inter_sesh_time * model_params["delta"]) * 3600
        cur_time += datetime.timedelta(seconds=scaled_intersesh_time)


def get_next_item(items, items_info, cur_time, next_new_item_idx, cached):
    """
    Finds the next item to be presented based on their activation.

    Arguments:
    items             -- the items to be considered when choosing the next item
    items_info        -- the map, containing each item's information
    cur_time          -- the time, at which the next item should be presented
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
    future_time = calc_future_time(cur_time)
    # recalculate each SEEN item's activation at future time with their updated alphas
    for item in seen_items:
        # Extract all encounters, which happened before current time
        prev_encounters   = [enc for enc in items_info[item].encounters if enc.time < cur_time]
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

    return next_item, next_new_item_idx_inc


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
    call_count  -- used to see how many recursive calls are made

    Returns:
    the activation of the item at the given timestamp
    """

    # NOTE: only here to count recursive function calls
    # stores the incremented call count when the function is called recursively
    call_count_inc = call_count

    # if there are NO previous encounters
    if len(encounters) == 0:
        m = np.NINF
    # ASSUMING that the encounters are sorted chronologically
    # if the last encounter happens later than the time of calculation
    elif encounters[len(encounters)-1].time > cur_time:
        raise ValueError("Encounters must happen BEFORE the time of activation calculation!")
    else:
        # stores the sum of time differences
        time_diff_sum = 0.0
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
            time_diff_sum += np.power(time_diff, -enc_decay)

        # calculate the activation given the sum of scaled time differences
        m = np.log(time_diff_sum)

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


def calc_reaction_time(activation):
    """
    Calculates the predicted reaction time (in seconds) based on an item's activation
    """
    reaction_time = 3.788 if np.isneginf(activation) else model_params["F"] * np.exp(np.negative(activation)) + model_params["f"]

    return reaction_time


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





# [ Main function ]

# def main():
    # learn_sesh_counts       = [50]
    # study_sesh_counts       = [2] # [4, 5, 7, 10]
    # study_sesh_lengths      = [1800] # [900, 1800, 2700, 3600]
    # immediate_alpha_adjusts = [True] #, False]
    # alpha_adjust_values     = [0.02]

    # # For all possible options for INTER-SESSION TIME
    # for inter_sesh_time in [24]: # [2, 4, 24]:
        # # For all possible options of USING the CACHED HISTORY
        # for cached in [True] # [False, True]:
            # # For all possible NUMBERS of LEARNING SESSIONS
            # for learn_sesh_count in learn_sesh_counts:
                # # For all possible NUMBERS of STUDY SESSIONS
                # for study_sesh_count in study_sesh_counts:
                    # # For all possible SESSION LENGTHS
                    # for study_sesh_length in study_sesh_lengths:
                        # # For all possible VALUES for ALPHA ADJUSTING
                        # for alpha_adjust_value in alpha_adjust_values:
                            # # For all possible WAYS of ADJUSTING the ALPHA
                            # for immediate_alpha_adjust in immediate_alpha_adjusts:
                                # # For all possible options of updating cached history
                                # for cache_update in ["", "immediately", "post-session"]:
                                    # # NOTE: these are here to avoid useless tests
                                    # # Since cache is NOT used, there is no use updating it
                                    # if not cached and cache_update != "":
                                        # continue
                                    # # Since alpha is adjusted POST-SESSION, there is no use updating the cache immediately
                                    # if not immediate_alpha_adjust and cache_update == "immediately":
                                        # continue

                                    # # alpha_adjust_value = 0.02 if immediate_alpha_adjust else 0.05

                                    # # NOTE: session LENGTHS are in SECONDS, while INTER-SESH time is in kOURS
                                    # reset_items_info(items_info)
                                    # print("\n\n\n")
                                    # print("Testing learning:")
                                    # print("Learn sessions         =", learn_sesh_count)
                                    # print("Inter-session time     =", inter_sesh_time)
                                    # print("Study sessions         =", study_sesh_count)
                                    # print("Study session length   =", study_sesh_length)
                                    # print("Immediate alpha adjust =", immediate_alpha_adjust)
                                    # print("Alpha adjust value     =", alpha_adjust_value)
                                    # print("Cache used             =", cached)
                                    # print("Cache updated          =", cache_update)
                                    # test_learning(items, items_info, study_sesh_count, study_sesh_length, learn_sesh_count, cached, immediate_alpha_adjust, alpha_adjust_value, cache_update, inter_sesh_time)



def main():
    test_encounter_history(items, items_info, 2, 1800, 24)

main()
