import steamapi
import csv
import json
import time
import datetime


def steam_search(steam_id_set, search_id_list, id_file_name, num_result):  
    count = 0
    # The result set may be larger than num_result because the search of one id has to be completed
    while count < num_result:
        # The first element in the search list is read and DELETED. see python "pop"
        parent = steamapi.user.SteamUser(search_id_list.pop(0))

        try: # In case friends is not opened
            fr = parent.friends
        except steamapi.errors.APIUnauthorized:
            continue

        for j in range(len(fr)):
            if fr[j].id not in steam_id_set:
                search_id_list.append(fr[j].id)
                steam_id_set.add(fr[j].id)

                count += 1
                if count%5000 == 0:
                    time.sleep(120)
                    print(count, "new results have been found")
    
    # write search_id_list to a csv file
    with open(id_file_name + "_search.csv", "w") as f:
        writer = csv.writer(f, delimiter ="\n")
        writer.writerow(search_id_list)

    # write steam_id_set to a csv file
    with open(id_file_name + ".csv", "w") as f:
        writer = csv.writer(f, delimiter ="\n")
        writer.writerow(steam_id_set)
            

def steam_info(steam_id_set, steam_info_data, id_file_name):
    """
    This function takes a set of steam ids and call steamapi to get the detailed information
    of them. The result is saved in a dicationary steam_info_data and in a json file.
    """
    count = 0
    for id in steam_id_set:
        # If this id has been searched already, ignore it (incase we want to increase datasize)
        if str(id) in steam_info_data.keys():
            continue

        # Call the required information
        user = steamapi.user.SteamUser(id)
        profile_open = True
        new_id_info = {}
        # If the user information is not accessable, return empty data. We would like it's 
        # key to remain in steam_info_data to prevent searching it again in the future.
        try:
            new_id_info["friends"] = [friend.id for friend in user.friends]
        except steamapi.errors.APIUnauthorized:
            new_id_info["friends"] = []
            new_id_info["games"] = []
            new_id_info["time_created"] = None
            new_id_info["level"] = None
            profile_open = False
        except steamapi.errors.AccessException:
            new_id_info["friends"] = [] 

        if profile_open: # The steam api has a bug that if API is not open, user.time_created does not exists
            try:
                new_id_info["games"] = user.game_time
            except steamapi.errors.AccessException:
                new_id_info["games"] = []
            except: # I don't know why this error is not captured by steamapi
                new_id_info["games"] = []
            try:
                date = user.time_created
                new_id_info["time_created"] = (date.year, date.month, date.day)
            except steamapi.errors.AccessException:
                new_id_info["time_created"] = None
            try:
                new_id_info["level"] = user.level
            except steamapi.errors.AccessException:
                new_id_info["level"] = None

        steam_info_data[id] = new_id_info

        # Saving during the process to prevent losing data
        count += 1
        if count%500 == 0:
            print(count, "IDs have been searched and saved, total", len(steam_info_data)) 
            with open(id_file_name + '.json', 'w') as f:
                json.dump(steam_info_data, f)
            if count%500 == 0:
                time.sleep(300)

    # Saving all the data
    with open(id_file_name + '.json', 'w') as f:
        json.dump(steam_info_data, f)


def game_info(steam_id_set, steam_info_data, id_file_name):
    """
    A reduced version of steam info, record only game info
    This function takes a set of steam ids and call steamapi to get the detailed information
    of them. The result is saved in a dicationary steam_info_data and in a json file.
    """
    count = 0
    for id in steam_id_set:
        # If this id has been searched already, ignore it (incase we want to increase datasize)
        if str(id) in steam_info_data.keys():
            continue

        # Call the required information
        user = steamapi.user.SteamUser(id)
        new_id_info = {}

        try:
            steam_info_data[id] = user.game_time
        except steamapi.errors.APIUnauthorized:
            steam_info_data[id] = []
        except:
            steam_info_data[id] = []

        # Saving during the process to prevent losing data
        count += 1
        if count%500 == 0:
            print(count, "IDs have been searched and saved, total", len(steam_info_data)) 
            with open(id_file_name + '.json', 'w') as f:
                json.dump(steam_info_data, f)
            if count%2000 == 0:
                time.sleep(180)

    # Saving all the data
    with open(id_file_name + '.json', 'w') as f:
        json.dump(steam_info_data, f)



def load_data(id_file_name):
    """
    If the data exist, read the existing search_id_list data, otherwise create one with the root id
    """
    try: 
        search_id_list = []     
        with open(id_file_name + "_search.csv", 'r') as f:
            reader = csv.reader(f, delimiter= '\n')
            for row in reader:
                try:
                    search_id_list.append(int(row[0]))
                except IndexError:
                    pass
        print("search_id_list is loaded from file")
    except FileNotFoundError:
        search_id_list = [root.id]
        print("search_id_list not found, a new one is created")

    try: 
        steam_id_set = set()
        with open(id_file_name + ".csv", 'r') as f:
            reader = csv.reader(f, delimiter = '\n')
            for row in reader:
                try:
                    steam_id_set.add(int(row[0]))
                except IndexError:
                    pass
        print("steam_id_set is loaded from file")
    except FileNotFoundError:
        steam_id_set = {root.id}
        print("steam_id_set not found, a new one is created")

    try: 
        json_data = open(id_file_name + ".json").read()
        steam_info_data = json.loads(json_data)
        print("steam_info_data is loaded from file")
    except FileNotFoundError:
        steam_info_data = {}
        print("steam_info_data not found, a new one is created")

    return steam_id_set, search_id_list, steam_info_data


# This seciton will not run if the py script is imported by another
steamapi.core.APIConnection(api_key="6B61866E0CAEBE0BE9804CAAB54502E9", validate_key=True)
root = steamapi.user.SteamUser(userurl="kane2019")

# parameters
id_file_name = "D://steamdata//user_game" # The file name where the data will be stored
data_size = 200000# The approxmated number of result in this round of search

# Important variables:
# steam_id_set: steam_id_set saves the result id, this is a set object, which is not ordered 
#               but elements can be efficiently searched. No duplication can exist in a set.
# search_id_list: search_id_list saves the id that whose friends remain to be searched. It is a list object.
#               All elements in this list already exists in steam_id_set
# steam_info_data: steam_info_data is a dictionary which uses the str(id) as key and contains information of 
#               the user. Currently the informaiton includes:
#               "games": a list of [game_id,play_time] (integer)
#               "friends": a list of friends id (integer)
#               "level"：an integer
#               "time_created": a tuple containing three integer (year, month, day)
#               CAREFUL: JSON only allow string to be the keys, thus the keys are always str(id) instead of an integer!
#

while(True):
    try:
        
        steam_id_set, search_id_list, steam_info_data = load_data(id_file_name)

        num_new_id = data_size-len(steam_id_set)
        steam_search(steam_id_set, search_id_list, id_file_name, num_new_id)

        game_info(steam_id_set, steam_info_data, id_file_name)

        if len(steam_info_data)>=1:
            break
    except:
        print("Connection error at", datetime.datetime.now())
        time.sleep(500)