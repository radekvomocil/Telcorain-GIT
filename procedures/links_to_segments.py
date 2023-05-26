import numpy as np
import math


def linear_repeat(calc_data, isecDic, segment_size):
    for cml in calc_data:
        if (((float(cml.site_a_longitude.data), float(cml.site_a_latitude.data)),
             (float(cml.site_b_longitude.data), float(cml.site_b_latitude.data)))) in isecDic:
            continue
        else:
            # Putting coordinates into variables
            SiteA = {"x": cml.site_a_longitude, "y": cml.site_a_latitude}
            SiteB = {"x": cml.site_b_longitude, "y": cml.site_b_latitude}

            distance: float = np.arccos(
                np.sin(SiteA["y"] * np.pi / 180) * np.sin(SiteB["y"] * np.pi / 180) + np.cos(
                    SiteA["y"] * np.pi / 180) * np.cos(SiteB["y"] * np.pi / 180) * np.cos(
                    SiteB["x"] * np.pi / 180 - SiteA["x"] * np.pi / 180)) * 6371000

            # Dividing link into 'x'm intervals
            if distance >= segment_size:
                numberOfPoints = distance / segment_size
            else:
                numberOfPoints = 2

            # Calculating gaps between each point in link
            gap_long = (SiteB["x"] - SiteA["x"]) / np.floor(numberOfPoints)
            gap_lat = (SiteB["y"] - SiteA["y"]) / np.floor(numberOfPoints)

            # Append into listOfSegments series of digits representing number of segments
            listOfSegments = []
            i = 1
            while i <= np.floor(numberOfPoints) + 1:
                listOfSegments.append(i)
                i += 1
            cml['segments'] = listOfSegments

            # Append coordinates of each point into lat_coords & long_coords
            long_coords = []
            lat_coords = []
            step = 0
            while step <= numberOfPoints:
                next_long_point = SiteA["x"] + gap_long * step
                next_lat_point = SiteA["y"] + gap_lat * step

                long_coords.append(next_long_point)
                lat_coords.append(next_lat_point)
                step += 1

            cml_data_id = []
            rr = 1
            while rr <= len(listOfSegments):
                cml_data_id.append(int(cml.cml_id.data))
                rr += 1

            cml['long_array'] = ('segments', long_coords)
            cml['lat_array'] = ('segments', lat_coords)
            cml['cml_reference'] = ('segments', cml_data_id)


def intersections(calc_data, isecDic):
    # list, do kterého se budou ukládat vzdálenosti jednotlivých křížení jednoho spoje
    distances = []
    for o in range(0, len(list(isecDic.values())[0]) - 1):
        distance = math.dist(list(isecDic.values())[0][o], list(isecDic.values())[0][o + 1])
        distances.append(distance)

    # list, do kterého se budou ukládat nejdelší úsečky křížení, přesněji teda souřadnice začátku a konce nejdelší úsečky daného spoje
    CoordsOfLongestLinesOfLinks = []

    # Vymyšlený algoritmus pro najití nejdelších úseček spojů, které se kříží
    for r in range(0, len(isecDic)):
        find_number_of_intersections = []
        listOfSegments_intersections = []
        long_coords_intersections = []
        lat_coords_intersections = []
        cml_references = []
        number = 0
        number += 1
        largestLine = max(distances)
        for j in range(0, len(distances)):
            if largestLine == distances[j]:
                print("Spoje pro nejdelší úsečku")
                print(f"Largest line is between points: {list(isecDic.values())[r][j]},->"
                      f"{list(isecDic.values())[r][j + 1]}")
                CoordsOfLongestLinesOfLinks.append(((list(isecDic.values())[r][j]),
                                                    (list(isecDic.values())[r][j + 1])))
                rain_values_for_longest_path_first_side = []
                rain_values_for_longest_path_second_side = []
                for q in range(0, len(isecDic)):
                    for w in range(0, len(list(isecDic.values())[q])):
                        if list(isecDic.values())[q][w] == list(isecDic.values())[r][j]:
                            for z in range(0, len(calc_data)):
                                if list(isecDic.values())[q][0][0] == calc_data[
                                    z].site_a_longitude.data and \
                                        list(isecDic.values())[q][0][1] == calc_data[
                                    z].site_a_latitude.data and \
                                        list(isecDic.values())[q][
                                            len(list(isecDic.values())[q]) - 1][0] \
                                        == calc_data[z].site_b_longitude.data and \
                                        list(isecDic.values())[q][
                                            len(list(isecDic.values())[q]) - 1][1] == calc_data[
                                    z].site_b_latitude.data:
                                    print(f"Našel se spoj: {list(isecDic)[q]} -> "
                                          f"{list(isecDic)[r]}")
                                    rain_values_for_longest_path_first_side.append(
                                        float(calc_data[z].R.mean().data))
                                    break
                                else:
                                    continue
                        else:
                            continue
                lowestRainValueForLongestPathFirstSide = min(
                    rain_values_for_longest_path_first_side)
                for u in range(0, len(isecDic)):
                    for d in range(0, len(list(isecDic.values())[u])):
                        if list(isecDic.values())[u][d] == list(isecDic.values())[r][j + 1]:
                            for s in range(0, len(calc_data)):
                                if list(isecDic.values())[u][0][0] == calc_data[
                                    s].site_a_longitude.data and \
                                        list(isecDic.values())[u][0][1] == calc_data[
                                    s].site_a_latitude.data and \
                                        list(isecDic.values())[u][
                                            len(list(isecDic.values())[u]) - 1][0] \
                                        == calc_data[s].site_b_longitude.data and \
                                        list(isecDic.values())[u][
                                            len(list(isecDic.values())[u]) - 1][1] \
                                        == calc_data[s].site_b_latitude.data:
                                    print(
                                        f"Našel/y se spoj/e druhého bodu křížení:{list(isecDic)[u]} -> "
                                        f"{list(isecDic)[r]}")
                                    rain_values_for_longest_path_second_side.append(
                                        float(calc_data[s].R.mean().data))
                                    break
                                else:
                                    continue
                        else:
                            continue
                lowestRainValueForLongestPathSecondSide = min(
                    rain_values_for_longest_path_second_side)

                if len(rain_values_for_longest_path_first_side) == 1 or len(
                        rain_values_for_longest_path_second_side) == 1:
                    middlepart = []
                    halfOfLongestLongitude = (list(isecDic.values())[r][j][0] +
                                              list(isecDic.values())[r][j + 1][0]) / 2
                    halfOfLongestLatitude = (list(isecDic.values())[r][j][1] +
                                             list(isecDic.values())[r][j + 1][1]) / 2
                    middlepart.append((halfOfLongestLongitude, halfOfLongestLatitude))
                    for c in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathFirstSide == float(
                                calc_data[c].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                print("levá <-> střed je, ale přeskočilo se, už je v listu souřadnice")
                                break
                            else:
                                print("levá <-> střed")
                                long_coords_intersections.append(list(isecDic.values())[r][j][0])
                                lat_coords_intersections.append(list(isecDic.values())[r][j][1])
                                find_number_of_intersections.append(number)
                                cml_references.append(int(calc_data[c].cml_id.data))
                                # TODO - Přidat přiřazení hodnoty (calc_data[c].R) pro spoj levá - střed
                                break
                        else:
                            continue
                    long_coords_intersections.append(halfOfLongestLongitude)
                    lat_coords_intersections.append(halfOfLongestLatitude)
                    find_number_of_intersections.append(number)
                    cml_references.append(int(calc_data[c].cml_id.data))
                    # TODO - Přiřadit menší z těch dvou

                    for b in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathSecondSide == float(
                                calc_data[b].R.mean().data):
                            print("střed <-> pravá")
                            long_coords_intersections.append(list(isecDic.values())[r][j + 1][0])
                            lat_coords_intersections.append(list(isecDic.values())[r][j + 1][1])
                            find_number_of_intersections.append(number)
                            cml_references.append(int(calc_data[b].cml_id.data))
                            # TODO - Přidat přiřazení hodnoty (calc_data[c].R) pro spoj střed - pravá
                            break
                        else:
                            continue
                else:
                    threeparts = []
                    firstThirdLongitude = abs(((list(isecDic.values())[r][j][0] -
                                            list(isecDic.values())[r][j + 1][0]) / 3) - list(isecDic.values())[r][j][0])
                    firstThirdLatitude = abs(((list(isecDic.values())[r][j][1] -
                                           list(isecDic.values())[r][j + 1][1]) / 3) - list(isecDic.values())[r][j][1])
                    secondThirdLongitude = abs(((list(isecDic.values())[r][j][0] -
                                             list(isecDic.values())[r][j + 1][0]) * 2 / 3) - \
                                           list(isecDic.values())[r][j][0])
                    secondThirdLatitude = abs(((list(isecDic.values())[r][j][1] -
                                            list(isecDic.values())[r][j + 1][1]) * 2 / 3) - \
                                           list(isecDic.values())[r][j][1])
                    threeparts.append((firstThirdLongitude, firstThirdLatitude))
                    threeparts.append((secondThirdLongitude, secondThirdLatitude))

                    for v in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathFirstSide == float(calc_data[v].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                print("levá <-> střed(levá) je, ale přeskočilo se, už je v listu souřadnice")
                                break
                            else:
                                print("levá <-> střed(levá)")
                                long_coords_intersections.append(list(isecDic.values())[r][j][0])
                                lat_coords_intersections.append(list(isecDic.values())[r][j][1])
                                find_number_of_intersections.append(number)
                                cml_references.append(int(calc_data[v].cml_id.data))
                                # TODO - Přidat přiřazení hodnoty (calc_data[c].R) pro spoj levá - střed(levá)
                                break
                        else:
                            continue
                    print("střed(levá) <-> střed(pravá)")
                    long_coords_intersections.append(firstThirdLongitude)
                    lat_coords_intersections.append(firstThirdLatitude)
                    find_number_of_intersections.append(number)
                    long_coords_intersections.append(secondThirdLongitude)
                    lat_coords_intersections.append(secondThirdLatitude)
                    find_number_of_intersections.append(number)
                    for gg in range(0, len(calc_data)):
                        if list(isecDic.values())[r][0][0] == calc_data[gg].site_a_longitude.data and \
                                list(isecDic.values())[r][0][1] == calc_data[gg].site_a_latitude.data and \
                                list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][0] \
                                == calc_data[gg].site_b_longitude.data and \
                                list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][1] == calc_data[
                            gg].site_b_latitude.data:
                            cml_references.append(int(calc_data[gg].cml_id.data))
                            cml_references.append(int(calc_data[gg].cml_id.data))
                        else:
                            continue
                    for n in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathSecondSide == float(
                                calc_data[n].R.mean().data):
                            print("střed(pravá) <-> pravá")
                            long_coords_intersections.append(list(isecDic.values())[r][j + 1][0])
                            lat_coords_intersections.append(list(isecDic.values())[r][j + 1][1])
                            find_number_of_intersections.append(number)
                            cml_references.append(int(calc_data[n].cml_id.data))
                            # TODO - Přidat přiřazení hodnoty (calc_data[c].R) pro spoj střed(pravá) - pravá
                            break
                        else:
                            continue
                    # TODO - Přidat přiřazení střední části - střed(levá) - střed(pravá) hodnotu původního spoje ([r][j]
            else:
                print("Spoje pro menší úseky:")
                print((f"Smaller line is between points: {list(isecDic.values())[r][j]},->"
                       f"{list(isecDic.values())[r][j + 1]}"))
                rain_values_for_shorter_path_first_side = []
                rain_values_for_shorter_path_second_side = []
                for f in range(0, len(isecDic)):
                    for g in range(0, len(list(isecDic.values())[f])):
                        if list(isecDic.values())[f][g] == list(isecDic.values())[r][j]:
                            for h in range(0, len(calc_data)):
                                if list(isecDic.values())[f][0][0] == calc_data[
                                    h].site_a_longitude.data and \
                                        list(isecDic.values())[f][0][1] == calc_data[
                                    h].site_a_latitude.data and \
                                        list(isecDic.values())[f][
                                            len(list(isecDic.values())[f]) - 1][0] == calc_data[
                                    h].site_b_longitude.data and \
                                        list(isecDic.values())[f][
                                            len(list(isecDic.values())[f]) - 1][1] == calc_data[
                                    h].site_b_latitude.data:
                                    print(f"Našel se spoj: {list(isecDic)[f]} -> "
                                          f"{list(isecDic)[r]}")
                                    rain_values_for_shorter_path_first_side.append(
                                        float(calc_data[h].R.mean().data))
                                    break
                                else:
                                    continue
                        else:
                            continue
                lowestRainValueForShorterPathFirstSide = min(
                    rain_values_for_shorter_path_first_side)
                for hh in range(0, len(isecDic)):
                    for y in range(0, len(list(isecDic.values())[hh])):
                        if list(isecDic.values())[hh][y] == list(isecDic.values())[r][j + 1]:
                            for k in range(0, len(calc_data)):
                                if list(isecDic.values())[hh][0][0] == calc_data[
                                    k].site_a_longitude.data and \
                                        list(isecDic.values())[hh][0][1] == calc_data[
                                    k].site_a_latitude.data and \
                                        list(isecDic.values())[hh][
                                            len(list(isecDic.values())[hh]) - 1][0] == \
                                        calc_data[k].site_b_longitude.data and \
                                        list(isecDic.values())[hh][
                                            len(list(isecDic.values())[hh]) - 1][1] == \
                                        calc_data[k].site_b_latitude.data:
                                    print(
                                        f"Našel/ly se spoj/e druhého bodu křížení:{list(isecDic)[hh]} -> "
                                        f"{list(isecDic)[r]}")
                                    rain_values_for_shorter_path_second_side.append(
                                        float(calc_data[h].R.mean().data))
                                    break
                                else:
                                    continue
                        else:
                            continue
                lowestRainValueForShorterPathSecondSide = min(
                    rain_values_for_shorter_path_second_side)
                if len(rain_values_for_shorter_path_first_side) == 1 or len(
                        rain_values_for_shorter_path_second_side) == 1:
                    lowestRainValue = min(lowestRainValueForShorterPathFirstSide,
                                          lowestRainValueForShorterPathSecondSide)
                    for m in range(0, len(calc_data)):
                        if lowestRainValue == float(calc_data[m].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                print(
                                    "Celý spoj je, ale přeskočilo se, už je v listu souřadnic levé křížení a přidalo se pouze pravé")
                                long_coords_intersections.append(list(isecDic.values())[r][j + 1][0])
                                lat_coords_intersections.append(list(isecDic.values())[r][j + 1][1])
                                find_number_of_intersections.append(number)
                                cml_references.append(int(calc_data[m].cml_id.data))
                                break
                            else:
                                print("Celý spoj")
                                long_coords_intersections.append(list(isecDic.values())[r][j][0])
                                lat_coords_intersections.append(list(isecDic.values())[r][j][1])
                                long_coords_intersections.append(list(isecDic.values())[r][j + 1][0])
                                lat_coords_intersections.append(list(isecDic.values())[r][j + 1][1])
                                find_number_of_intersections.append(number)
                                find_number_of_intersections.append(number)
                                cml_references.append(int(calc_data[m].cml_id.data))
                                cml_references.append(int(calc_data[m].cml_id.data))
                                break
                                # TODO - Přidat přiřazení hodnoty (calc_data[c].R) pro celý spoj
                else:
                    middlepart_short_path = []
                    halfOfShorterPathLongitude = (list(isecDic.values())[r][j][0] +
                                                  list(isecDic.values())[r][j + 1][0]) / 2
                    halfOfShorterPathLatitude = (list(isecDic.values())[r][j][1] +
                                                 list(isecDic.values())[r][j + 1][1]) / 2
                    middlepart_short_path.append(
                        (halfOfShorterPathLongitude, halfOfShorterPathLatitude))
                    for qq in range(0, len(calc_data)):
                        if lowestRainValueForShorterPathFirstSide == float(
                                calc_data[qq].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                print("levá <-> střed je, ale přeskočilo se, už je v listu souřadnice")
                                break
                            else:
                                print("levá <-> střed")
                                long_coords_intersections.append(list(isecDic.values())[r][j][0])
                                lat_coords_intersections.append(list(isecDic.values())[r][j][1])
                                find_number_of_intersections.append(number)
                                cml_references.append(int(calc_data[qq].cml_id.data))
                                # TODO - Přidat přiřazení hodnoty (calc_data[c].R) pro spoj levá - střed
                                break
                        else:
                            continue
                    long_coords_intersections.append(halfOfShorterPathLongitude)
                    lat_coords_intersections.append(halfOfShorterPathLatitude)
                    find_number_of_intersections.append(number)
                    cml_references.append(int(calc_data[qq].cml_id.data))
                    for ww in range(0, len(calc_data)):
                        if lowestRainValueForShorterPathSecondSide == float(
                                calc_data[ww].R.mean().data):
                            print("střed <-> pravá")
                            long_coords_intersections.append(list(isecDic.values())[r][j + 1][0])
                            lat_coords_intersections.append(list(isecDic.values())[r][j + 1][1])
                            find_number_of_intersections.append(number)
                            cml_references.append(int(calc_data[ww].cml_id.data))
                            break
                        else:
                            continue
        sections = 1
        while sections <= len(find_number_of_intersections):
            listOfSegments_intersections.append(sections)
            sections += 1
        print(find_number_of_intersections)
        print(long_coords_intersections)
        print(lat_coords_intersections)
        print(cml_references)
        print(listOfSegments_intersections)
        print("\n")
        current_cml = None
        for spoj in range(0, len(calc_data)):
            if list(isecDic.values())[r][0][0] == calc_data[spoj].site_a_longitude.data and \
                    list(isecDic.values())[r][0][1] == calc_data[spoj].site_a_latitude.data and \
                    list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][0] \
                    == calc_data[spoj].site_b_longitude.data and \
                    list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][1] == calc_data[
                spoj].site_b_latitude.data:
                current_cml = calc_data[spoj]
                break
            else:
                continue
        current_cml['segments'] = listOfSegments_intersections
        current_cml['long_array'] = ('segments', long_coords_intersections)
        current_cml['lat_array'] = ('segments', lat_coords_intersections)
        current_cml['cml_reference'] = ('segments', cml_references)
        distances = []

        if r == len(isecDic) - 1:
            continue
        else:
            for oo in range(0, len(list(isecDic.values())[(r + 1)]) - 1):
                distance = math.dist(list(isecDic.values())[(r + 1)][oo],
                                     list(isecDic.values())[(r + 1)][oo + 1])
                distances.append(distance)
