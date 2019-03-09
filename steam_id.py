def NextLevel(idp, id_parent):
    steamid_son = []
    for i in range(len(id_parent)):
        parent = steamapi.user.SteamUser(id_parent[i])
        try:
            fr = parent.friends
            n=len(idp)
            for j in range(len(fr)):
                ind = 0
                for k in range(n):
                    if fr[j].id == idp[k]:
                        ind = ind+1
                if ind == 0:
                    steamid_son.append(fr[j].id)
                    idp.append(fr[j].id)
        except:
            continue
    return [steamid_son, idp]
def main():
    steam_id = [[me.id]]
    idp = [me.id]
    for i_o in range(3):
        ex = NextLevel(idp, steam_id[i_o])
        idp = ex[1]
        steam_id.append(ex[0])
