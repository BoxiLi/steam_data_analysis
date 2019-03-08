def NextLevel(id_parent):
    steamid_son = []
    for i in range(len(id_parent)):
        n=0
        parent = steamapi.user.SteamUser(id_parent[i])
        try:
            fr = parent.friends
            for j in range(len(fr)):
                steamid_son.append(fr[j].id)
                j = j+1
        except:
            continue
    return steamid_son
def main():
    steam_id = [[me.id]]
    for i_o in range(3):
        steam_id.append(NextLevel(steam_id[i_o]))
