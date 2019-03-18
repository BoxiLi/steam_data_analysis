import steamapi
import pandas as pd
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
            for j in range(len(fr)):
                if fr[j].id not in steam_id_set:
                    steam_id_set.add(fr[j].id)
                    search_id_list.append(fr[j].id)
            
                    # game infomation can be added here

                    count += 1
                    if count%1000 == 0:
                        print(count, "new results have been found")
        except steamapi.errors.APIUnauthorized as e:
            continue
    
    # write search_id_list to a csv file
    with open(id_file_name + "_search.csv", "w") as f:
        writer = csv.writer(f, delimiter ="\n")
        writer.writerow(search_id_list)

    # write steam_id_set to a csv file
    # can this be done without a new copy?
    id_data = pd.DataFrame(list(steam_id_set), columns=['id'])
    id_data.to_csv(id_file_name+".csv", index=False)
            



# This seciton will not run if the py script is imported by another
steamapi.core.APIConnection(api_key="6B61866E0CAEBE0BE9804CAAB54502E9", validate_key=True)
root = steamapi.user.SteamUser(userurl="kane2019")

# parameters
id_file_name = "test" # The file name where the data will be stored
num_result = 50 # The approxmated number of result of result in this search

# Important variables:
# steam_id_set: steam_id_set saves the result id, this is a set object, which is not ordered 
#               but elements can be efficiently searched. No duplication can exist in a set.
# search_id_list: search_id_list saves the id that whose friends remain to be searched. It is a list object.
#               All elements in this list already exists in steam_id_set

# If the data exist, read the existing search_id_list data, otherwise create one with the root id
try: 
    search_id_list = []     
    with open(id_file_name + "_search.csv", 'r') as f:
        reader = csv.reader(f, delimiter= "\n")
        try:
            for row in reader:
                search_id_list.append(int(row[0]))
        except IndexError:
            pass
    id_data = pd.read_csv(id_file_name + ".csv")
    steam_id_set = set(id_data["id"].tolist())

except:
    steam_id_set = {root.id}
    search_id_list = [root.id]
steam_search(steam_id_set, search_id_list, id_file_name, num_result)
    
