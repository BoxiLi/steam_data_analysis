import steamapi
import csv
import json

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
                if count%1000 == 0:
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
    This function take a set of steam id and call steamapi to get the detailed information
    of them. The result is saved in a dicationary steam_info_data and in a json file.
    """
    count = 0
    for id in steam_id_set:
        # If this id has been searched already, ignore it (incase we want to increase datasize)
        if str(id) in steam_info_data.keys():
            continue

        # Call the required information
        user = steamapi.user.SteamUser(id)
        try:
            friends = [friend.id for friend in user.friends]
            games = [game.id for game in user.games] # we save only the integer id
            date = user.time_created
            time_created = (date.year, date.month, date.day)
            level = user.level
        except (steamapi.errors.APIUnauthorized, steamapi.errors.AccessException):
            # If the user information is not accessable, return empty data. We would like it's 
            # key to remain in steam_info_data to prevent searching it again in the future.
            friends = [] 
            games = []
            time_created = None
            level = None
        

        steam_info_data[id] = {"games":games, "friends":friends, "time_created":time_created,
            "level":level}

        # Saving during the process to prevent losing data
        count += 1
        if count%100 == 0:
            print(count, "IDs have been searched and saved")
            with open(id_file_name + '.json', 'w') as f:
                json.dump(steam_info_data, f)

    # Saving all the data
    with open(id_file_name + '.json', 'w') as f:
        json.dump(steam_info_data, f)



# This seciton will not run if the py script is imported by another
steamapi.core.APIConnection(api_key="6B61866E0CAEBE0BE9804CAAB54502E9", validate_key=True)
root = steamapi.user.SteamUser(userurl="kane2019")

# parameters
id_file_name = "test" # The file name where the data will be stored
num_result = 5 # The approxmated number of result of result in this search

# Important variables:
# steam_id_set: steam_id_set saves the result id, this is a set object, which is not ordered 
#               but elements can be efficiently searched. No duplication can exist in a set.
# search_id_list: search_id_list saves the id that whose friends remain to be searched. It is a list object.
#               All elements in this list already exists in steam_id_set
# steam_info_data: steam_info_data is a dictionary which uses the str(id) as key and contains information of 
#               the user. Currently the informaiton includes:
#               "games": a list of game id (integer)
#               "friends": a list of friends id (integer)
#               "level"：an integer
#               "time_created": a tuple containing three integer (year, month, day)
#               CAREFUL: JSON only allow string to be the keys, thus the keys are always str(id) instead of an integer!
#

# If the data exist, read the existing search_id_list data, otherwise create one with the root id
try: 
    search_id_list = []     
    with open(id_file_name + "_search.csv", 'r') as f:
        reader = csv.reader(f, delimiter= '\n')
        for row in reader:
            try:
                search_id_list.append(int(row[0]))
            except IndexError:
                pass

    steam_id_set = set()
    with open(id_file_name + ".csv", 'r') as f:
        reader = csv.reader(f, delimiter = '\n')

        for row in reader:
            try:
                steam_id_set.add(int(row[0]))
            except IndexError:
                pass

    json_data = open(id_file_name + ".json").read()
    steam_info_data = json.loads(json_data)

except FileNotFoundError:
    steam_id_set = {root.id}
    search_id_list = [root.id]
    steam_info_data = {}

steam_search(steam_id_set, search_id_list, id_file_name, num_result)

steam_info(steam_id_set, steam_info_data, id_file_name)
