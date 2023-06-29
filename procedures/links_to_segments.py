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

    # list, do kterého se budou ukládat nejdelší úsečky křížení, přesněji teda souřadnice začátku a konce
    # nejdelší úsečky daného spoje
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
                CoordsOfLongestLinesOfLinks.append(((list(isecDic.values())[r][j]),
                                                    (list(isecDic.values())[r][j + 1])))

                rain_values_for_longest_path_first_side = []
                rain_values_for_longest_path_second_side = []

                for_cycle(isecDic, calc_data, rain_values_for_longest_path_first_side, list(isecDic.values())[r][j])
                lowestRainValueForLongestPathFirstSide = min(rain_values_for_longest_path_first_side)

                for_cycle(isecDic, calc_data, rain_values_for_longest_path_second_side,
                          list(isecDic.values())[r][j + 1])
                lowestRainValueForLongestPathSecondSide = min(rain_values_for_longest_path_second_side)

                if len(rain_values_for_longest_path_first_side) == 1 or len(
                        rain_values_for_longest_path_second_side) == 1:
                    middlepart = []
                    halfOfLongestLongitude = (list(isecDic.values())[r][j][0] +
                                              list(isecDic.values())[r][j + 1][0]) / 2
                    halfOfLongestLatitude = (list(isecDic.values())[r][j][1] +
                                             list(isecDic.values())[r][j + 1][1]) / 2
                    middlepart.append((halfOfLongestLongitude, halfOfLongestLatitude))

                    c = None
                    for c in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathFirstSide == float(
                                calc_data[c].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                break
                            else:
                                appending(long_coords_intersections, list(isecDic.values())[r][j][0],
                                          lat_coords_intersections, list(isecDic.values())[r][j][1],
                                          find_number_of_intersections, number, cml_references,
                                          int(calc_data[c].cml_id.data))
                                break
                        else:
                            continue

                    appending(long_coords_intersections, halfOfLongestLongitude, lat_coords_intersections,
                              halfOfLongestLatitude, find_number_of_intersections, number, cml_references,
                              int(calc_data[c].cml_id.data))

                    for b in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathSecondSide == float(
                                calc_data[b].R.mean().data):
                            appending(long_coords_intersections, list(isecDic.values())[r][j + 1][0],
                                      lat_coords_intersections, list(isecDic.values())[r][j + 1][1],
                                      find_number_of_intersections, number, cml_references,
                                      int(calc_data[b].cml_id.data))
                            break
                        else:
                            continue
                else:
                    threeparts = []
                    firstThirdLongitude = abs(((list(isecDic.values())[r][j][0] -
                                                list(isecDic.values())[r][j + 1][0]) / 3) -
                                              list(isecDic.values())[r][j][0])
                    firstThirdLatitude = abs(((list(isecDic.values())[r][j][1] -
                                               list(isecDic.values())[r][j + 1][1]) / 3) - list(isecDic.values())[r][j][
                                                 1])
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
                                break
                            else:
                                appending(long_coords_intersections, list(isecDic.values())[r][j][0],
                                          lat_coords_intersections, list(isecDic.values())[r][j][1],
                                          find_number_of_intersections, number, cml_references,
                                          int(calc_data[v].cml_id.data))
                                break
                        else:
                            continue

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
                                list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][1] == calc_data[gg]. \
                                site_b_latitude.data:
                            cml_references.append(int(calc_data[gg].cml_id.data))
                            cml_references.append(int(calc_data[gg].cml_id.data))
                        else:
                            continue

                    for n in range(0, len(calc_data)):
                        if lowestRainValueForLongestPathSecondSide == float(
                                calc_data[n].R.mean().data):
                            appending(long_coords_intersections, list(isecDic.values())[r][j + 1][0],
                                      lat_coords_intersections, list(isecDic.values())[r][j + 1][1],
                                      find_number_of_intersections, number, cml_references,
                                      int(calc_data[n].cml_id.data))
                            break
                        else:
                            continue
            else:
                rain_values_for_shorter_path_first_side = []
                rain_values_for_shorter_path_second_side = []

                for_cycle(isecDic, calc_data, rain_values_for_shorter_path_first_side, list(isecDic.values())[r][j])
                lowestRainValueForShorterPathFirstSide = min(rain_values_for_shorter_path_first_side)

                for_cycle(isecDic, calc_data, rain_values_for_shorter_path_second_side,
                          list(isecDic.values())[r][j + 1])
                lowestRainValueForShorterPathSecondSide = min(rain_values_for_shorter_path_second_side)

                if len(rain_values_for_shorter_path_first_side) == 1 or len(
                        rain_values_for_shorter_path_second_side) == 1:
                    lowestRainValue = min(lowestRainValueForShorterPathFirstSide,
                                          lowestRainValueForShorterPathSecondSide)
                    for m in range(0, len(calc_data)):
                        if lowestRainValue == float(calc_data[m].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                appending(long_coords_intersections, list(isecDic.values())[r][j + 1][0],
                                          lat_coords_intersections, list(isecDic.values())[r][j + 1][1],
                                          find_number_of_intersections, number, cml_references,
                                          int(calc_data[m].cml_id.data))
                                break
                            else:
                                appending(long_coords_intersections, list(isecDic.values())[r][j][0],
                                          lat_coords_intersections, list(isecDic.values())[r][j][1],
                                          find_number_of_intersections, number, cml_references,
                                          int(calc_data[m].cml_id.data))

                                appending(long_coords_intersections, list(isecDic.values())[r][j + 1][0],
                                          lat_coords_intersections, list(isecDic.values())[r][j + 1][1],
                                          find_number_of_intersections, number, cml_references,
                                          int(calc_data[m].cml_id.data))
                                break
                else:
                    middlepart_short_path = []
                    halfOfShorterPathLongitude = (list(isecDic.values())[r][j][0] +
                                                  list(isecDic.values())[r][j + 1][0]) / 2
                    halfOfShorterPathLatitude = (list(isecDic.values())[r][j][1] +
                                                 list(isecDic.values())[r][j + 1][1]) / 2
                    middlepart_short_path.append(
                        (halfOfShorterPathLongitude, halfOfShorterPathLatitude))

                    qq = None
                    for qq in range(0, len(calc_data)):
                        if lowestRainValueForShorterPathFirstSide == float(
                                calc_data[qq].R.mean().data):
                            if list(isecDic.values())[r][j][0] in long_coords_intersections:
                                break
                            else:
                                appending(long_coords_intersections, list(isecDic.values())[r][j][0],
                                          lat_coords_intersections, list(isecDic.values())[r][j][1],
                                          find_number_of_intersections, number, cml_references,
                                          int(calc_data[qq].cml_id.data))
                                break
                        else:
                            continue

                    appending(long_coords_intersections, halfOfShorterPathLongitude, lat_coords_intersections,
                              halfOfShorterPathLatitude, find_number_of_intersections, number, cml_references,
                              int(calc_data[qq].cml_id.data))

                    for ww in range(0, len(calc_data)):
                        if lowestRainValueForShorterPathSecondSide == float(
                                calc_data[ww].R.mean().data):
                            appending(long_coords_intersections, list(isecDic.values())[r][j + 1][0],
                                      lat_coords_intersections, list(isecDic.values())[r][j + 1][1],
                                      find_number_of_intersections, number, cml_references,
                                      int(calc_data[ww].cml_id.data))
                            break
                        else:
                            continue
        sections = 1
        while sections <= len(find_number_of_intersections):
            listOfSegments_intersections.append(sections)
            sections += 1

        current_cml = None

        for spoj in range(0, len(calc_data)):
            if list(isecDic.values())[r][0][0] == calc_data[spoj].site_a_longitude.data and \
                    list(isecDic.values())[r][0][1] == calc_data[spoj].site_a_latitude.data and \
                    list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][0] \
                    == calc_data[spoj].site_b_longitude.data and \
                    list(isecDic.values())[r][len(list(isecDic.values())[r]) - 1][1] == \
                    calc_data[spoj].site_b_latitude.data:
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


def appending(long_intersections, coordinates, lat_intersections, coordinates_1, number_of_intersections, number,
              references, cml_data):
    long_intersections.append(coordinates)
    lat_intersections.append(coordinates_1)
    number_of_intersections.append(number)
    references.append(cml_data)


def for_cycle(isecDic, calc_data, rain_values_side, side_coords):
    for q in range(0, len(isecDic)):
        for w in range(0, len(list(isecDic.values())[q])):
            if list(isecDic.values())[q][w] == side_coords:
                for z in range(0, len(calc_data)):
                    if list(isecDic.values())[q][0][0] == calc_data[z].site_a_longitude.data and \
                            list(isecDic.values())[q][0][1] == calc_data[z].site_a_latitude.data and \
                            list(isecDic.values())[q][len(list(isecDic.values())[q]) - 1][0] \
                            == calc_data[z].site_b_longitude.data and \
                            list(isecDic.values())[q][len(list(isecDic.values())[q]) - 1][1] \
                            == calc_data[z].site_b_latitude.data:
                        rain_values_side.append(
                            float(calc_data[z].R.mean().data))
                        break
                    else:
                        continue
            else:
                continue
