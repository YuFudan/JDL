def calculate_travel_distance(action):
    if action["type"] == "在楼间移动":
        poly1, poly2 = buildings[action["start_building"]]["poly"], buildings[action["end_building"]]["poly"]
        return poly1.centroid.distance(poly2.centroid) * 2**0.5
    elif action["type"] == "楼外某点到楼":
        p, poly = action["start_gps"], buildings[action["end_building"]]["poly"]
        return Point(projector(*p)).distance(poly.centroid) * 2**0.5
    elif action["type"] == "楼到楼外某点":
        poly, p = buildings[action["start_building"]]["poly"], action["end_gps"]
        return Point(projector(*p)).distance(poly.centroid) * 2**0.5
    else:
        print("this is not a moving action!")
        return 0


def calculate_work_time(actions):
    h2workt = {h:0 for h in range(24)}
    for action in actions:
        if action["type"] in ["在楼间移动", "在楼内"]:
            ts, te = action["start_time"], action["end_time"]
            hs = ts // 3600
            ss = ts - hs * 3600
            he = te // 3600
            se = te - he * 3600
            if hs == he:
                h2workt[hs] += se - ss
            elif he > hs:
                for h in range(hs, he + 1):
                    if h == hs:
                        h2workt[h] += 3600 - ss
                    elif h == he:
                        h2workt[h] += se
                    else:
                        h2workt[h] += 3600
    return h2workt


def plot_cdf(h2num):
    """
    输入每个小时的统计量dict{0:10, 1:10, ...}
    画截止每个小时(把之前所有小时的数全加上{0:10, 1:20, ...})的CDF图
    """
    hs = sorted(list(h2num.keys()))
    cnums = [sum([h2num[h] for h in range(hh+1)]) for hh in range(24)]
    plt.figure(figsize=(12, 5))
    plt.xlim((0, hs[-1]+1.5))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    # plt.bar(hs, cnums, width=1.0)
    x = []
    for h in hs + [hs[-1]+1]:
        x.append(h)
        if hs[0] < h:
            x.append(h)
    y = [0, 0]
    for n in cnums:
        y.append(n)
        y.append(n)
    y = y[:-1]
    plt.plot(x, y)

    x = hs + [hs[-1]+1]
    y = [0] + cnums
    plt.plot(x, y)
    plt.savefig("figure/test.png")
