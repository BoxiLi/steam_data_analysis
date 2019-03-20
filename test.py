import steamapi
import json
id_file_name = "test"
json_data = open(id_file_name + ".json").read()
steam_info_data = json.loads(json_data)

for id in steam_info_data:
    if len(steam_info_data[id]["friends"])==0:
        miss_id = int(id)
        user = steamapi.user.SteamUser(miss_id)
        try:
            user.friends[0]
            print("here")

        except:
            pass
    