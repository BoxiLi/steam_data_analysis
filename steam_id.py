import steamapi

def steam_search(steam_id_set, search_id_list):
    count = 0
    while count < 1500:
        # The first element in the search list is read and DELETED. see python "pop"
        parent = steamapi.user.SteamUser(search_id_list.pop(0))

        try: # In case friends is not opened
            fr = parent.friends
            for j in range(len(fr)):
                if fr[j] not in steam_id_set:
                    count += 1
                    steam_id_set.add(fr[j].id)
                    search_id_list.append(fr[j].id)

            if count%1000 == 0:
                print(count, "new results have been found")

        except:
            continue
            
            # game infomation can be added here



if __name__ == "__main__":
    # This seciton will not run if the py script is imported by another

    # The root ID
    root_id = "6B61866E0CAEBE0BE9804CAAB54502E9"
    
    # The following two variable can later also be imported from a csv file

    # steam_id_set saves the result id, this is a set object, which is not ordered 
    # but elements can be efficiently searched. No duplication can exist in a set.
    steam_id_set = {root_id}
    # steam_id_list saves the id that whose friends remain to be searched. It is a list object.
    search_id_list = [root_id]

    steam_search(steam_id_set, search_id_list)
    
