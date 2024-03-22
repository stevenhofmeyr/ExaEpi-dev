#!/usr/bin/env python

import sys
import os.path
import struct
import pandas
from pandas.api.types import CategoricalDtype
import numpy as np
import time
import argparse
import geopandas


# note: missing category indexes are written as -1
hh_type = CategoricalDtype(categories=["hh", "gq"])
hh_living_arrangement = CategoricalDtype(categories=["married", "male_no_spouse", "female_no_spouse", "alone", "not_alone"])
hh_has_kids = CategoricalDtype(categories=["no", "yes"])
hh_dwg = CategoricalDtype(categories=["single_fam_detach", "single_fam_attach", "2_unit", "3_4_unit", "5_9_unit",
                                      "10_19_unit", "20_49_unit", "GE50_unit", "mob_home", "other"])
hh_tenure = CategoricalDtype(categories=["own", "rent", "other"])
hh_vehicles = CategoricalDtype(categories=["01", "02", "03", "04", "05", "GE06"])
pr_sex = CategoricalDtype(categories=["female", "male"])
pr_race = CategoricalDtype(categories=["white", "blk_af_amer", "asian", "native_amer", "pac_island", "other", "mult"])
pr_hsplat = CategoricalDtype(categories=["no", "yes"])
pr_ipr = CategoricalDtype(categories=["L050", "050_099", "100_124", "125_149", "150_184", "185_199", "GE200"])
pr_emp_stat = CategoricalDtype(categories=["not.in.force", "unemp", "employed", "mil"])
pr_travel = CategoricalDtype(categories=["car_truck_van", "public_transportation", "bicycle", "walked", "motorcycle", "taxicab",
                                          "other", "wfh"])
pr_veh_occ = CategoricalDtype(categories=["drove_alone", "carpooled"])
pr_grade = CategoricalDtype(categories=["preschl", "kind", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
                                         "11th", "12th", "undergrad", "grad"])
PUMS_ID_LEN = 13
NAICS_LEN = 8

def print_header(df):
    hdr_fname = "UrbanPopAgentStruct.H"
    print("Writing C++ header file at", hdr_fname)
    # print out header file for C++ import
    f_hdr = open(hdr_fname, "w")
    print("#include <stdlib.h>\n", file=f_hdr)
    print("#include <fstream>\n", file=f_hdr)
    print("struct UrbanPopAgent {", file=f_hdr)

    read_func_str = ""
    write_str = ""
    check_str = ""

    for i in range(len(df.columns)):
        c_type = str(df.dtypes[i]) + "_t"
        if c_type.startswith("float"):
            c_type = "float"

        if df.columns[i] == 'pums_id':
            print("    char " + df.columns[i] + "[" + str(PUMS_ID_LEN) + "];", file=f_hdr)
            write_str += "        os << \"b'\" << std::string(agent." + df.columns[i] + ", " + str(PUMS_ID_LEN) + ") << \"',\";\n"
        elif df.columns[i] == 'pr_naics':
            print("    char " + df.columns[i] + "[" + str(NAICS_LEN) + "];", file=f_hdr)
            write_str += "        os << \"b\'\" << std::string(agent." + df.columns[i] + ", " + str(NAICS_LEN) + ") << \"'\";\n"
        else:
            print("    " + c_type + " " + df.columns[i] + ";", file=f_hdr)
        if df.columns[i] in ["pums_id", "pr_naics"]:
            read_func_str += "        f.read(" + df.columns[i] + ", sizeof(" + df.columns[i] + "));\n"
        else:
            read_func_str += "        f.read((char*)&" + df.columns[i] + ", sizeof(" + df.columns[i] + "));\n"
            if c_type == "int8_t":
                c_type = "int32_t"
            write_str += "        os << (" + c_type + ")agent." + df.columns[i] + " << ',';\n"
            check_str += "        if (" + df.columns[i] + " != -99) " + \
                         "{std::cerr << \"Error: invalid first record for " + df.columns[i] + " \" << (" + c_type + \
                         ")" + df.columns[i] + \
                         " << std::endl; abort();}\n"

    print("\n    void read_binary(std::ifstream &f) {", file=f_hdr)
    print(read_func_str, file=f_hdr, end='')
    print("    }\n", file=f_hdr)

    print("\n    friend std::ostream& operator<<(std::ostream& os, const UrbanPopAgent& agent) {", file=f_hdr)
    print(write_str, file=f_hdr, end="")
    print("        return os;", file=f_hdr)
    print("    }", file=f_hdr)

    print("\n    void check_binary_inputs() {", file=f_hdr)
    print(check_str, file=f_hdr, end="")
    print("    }", file=f_hdr)

    print("};", file=f_hdr)

    print("Wrote", len(df.columns), "fields to", hdr_fname)


def process_feather_file(fname, fname_bin, geoid_locs_map, first):
    print("Reading data from", fname)
    t = time.time()
    df = pandas.read_feather(fname)
    print("Read", len(df.index), "records in %.3f s" % (time.time() - t))

    # all the fields parsed, in order
    df.p_id = df.p_id.str.split("-").str[-1].astype("int32")
    df.pums_id = df.pums_id.map(lambda x: str(x).ljust(PUMS_ID_LEN)).str.encode("utf-8")#.astype("string")
    max_pums_id_len = df.pums_id.map(len).max()
    if max_pums_id_len != PUMS_ID_LEN:
        print("Incorrect max PUMS_ID length", max_pums_id_len, "!=", PUMS_ID_LEN, file=sys.stderr)
    df.h_id = df.h_id.str.split("-").str[-1].astype("int32")
    df.geoid = df.geoid.astype("int64")
    df.hh_size = df.hh_size.astype("int8")
    df.hh_type = df.hh_type.astype(hh_type).cat.codes
    df.hh_living_arrangement = df.hh_living_arrangement.astype(hh_living_arrangement).cat.codes
    df.hh_age = df.hh_age.astype("int8")
    df.hh_has_kids = df.hh_has_kids.astype(hh_has_kids).cat.codes
    df.hh_income = df.hh_income.astype("int32")
    df.hh_nb_wrks = df.hh_nb_wrks.astype("int8")
    df.hh_nb_non_wrks = df.hh_nb_non_wrks.astype("int8")
    df.hh_nb_adult_wrks = df.hh_nb_adult_wrks.astype("int8")
    df.hh_nb_adult_non_wrks = df.hh_nb_adult_non_wrks.astype("int8")
    df.hh_dwg = df.hh_dwg.astype(hh_dwg).cat.codes
    df.hh_tenure = df.hh_tenure.astype(hh_tenure).cat.codes
    df.hh_vehicles = df.hh_vehicles.astype(hh_vehicles).cat.codes
    df.pr_age = df.pr_age.astype("int8")
    df.pr_sex = df.pr_sex.astype(pr_sex).cat.codes
    df.pr_race = df.pr_race.astype(pr_race).cat.codes
    df.pr_hsplat = df.pr_hsplat.astype(pr_hsplat).cat.codes
    df.pr_ipr = df.pr_ipr.astype(pr_ipr).cat.codes
    df.pr_naics = df.pr_naics.map(lambda x: str(x).ljust(NAICS_LEN)).str.encode("utf-8")#.astype("string")
    max_naics_len = df.pr_naics.map(len).max()
    if max_naics_len != NAICS_LEN:
        print("Incorrect max PR_NAICS length", max_naics_len, "!=", NAICS_LEN, file=sys.stderr)
    df.pr_emp_stat = df.pr_emp_stat.astype(pr_emp_stat).cat.codes
    df.pr_travel = df.pr_travel.astype(pr_travel).cat.codes
    df.pr_veh_occ = df.pr_veh_occ.astype(pr_veh_occ).cat.codes
    df.pr_commute = df.pr_commute.astype("int8")
    df.pr_grade = df.pr_grade.astype(pr_grade).cat.codes

    fname, _ = os.path.splitext(fname)
    fname_csv = fname + ".csv"

    #print("Unique PUMS", len(df.pums_id.unique()), "max length", df.pums_id.map(len).max())
    #print("Unique NAICS", len(df.pr_naics.unique()), "max length", df.pr_naics.map(len).max())

    # pack structure by moving int32_t value to before all int8_t values
    df.insert(4, "hh_income", df.pop("hh_income"))

    # move char arrays to end of struct
    df.insert(len(df.columns) - 1, "pums_id", df.pop("pums_id"))
    df.insert(len(df.columns) - 1, "pr_naics", df.pop("pr_naics"))

    # add lat/long locations from geoids
    df.insert(df.columns.get_loc("geoid") + 1, "latitude", float(0))
    df.latitude = df.latitude.astype("float32")
    df.insert(df.columns.get_loc("latitude") + 1, "longitude", float(0))
    df.longitude = df.longitude.astype("float32")

    if first:
        print_header(df)

    t = time.time()
    print("Setting lat/long for data", end=" ", flush=True)
    # find lat/long for each row entry
    df["latitude"] = df["geoid"].map(geoid_locs_map).apply(lambda x: x[0]).astype("float32")
    df["longitude"] = df["geoid"].map(geoid_locs_map).apply(lambda x: x[1]).astype("float32")
    print("\nSet lat/long for", len(df.index), "agents in %.3f s" % (time.time() - t))

    fmt = ""
    for i in range(len(df.columns)):
        if df.columns[i] == 'pums_id':
            fmt += str(PUMS_ID_LEN) + 's'
        elif df.columns[i] == 'pr_naics':
            fmt += str(NAICS_LEN) + 's'
        elif df.dtypes[i] == np.int8:
            fmt += 'b'
        elif df.dtypes[i] == np.int32:
            fmt += 'i'
        elif df.dtypes[i] == np.int64:
            fmt += 'q'
        elif df.dtypes[i] == np.float32:
            fmt += 'f'
        else:
            raise RuntimeError("ERRROR: dtype", df.dtypes[i], "in column", df.columns[i], "doesn't match a format string")

    # save the dtypes
    correct_dtypes = df.dtypes

    if first:
        check_settings = [-99] * (len(df.columns) - 2)
        check_settings.extend([b'-' * PUMS_ID_LEN, b'-' * NAICS_LEN])
        #print(check_settings)
        check_record = pandas.DataFrame([check_settings], columns=df.columns)
        # coerce new record to correct types
        check_record = check_record.astype(correct_dtypes)
        df = pandas.concat([check_record, df], ignore_index=True)
        print("Fields are:\n", df.dtypes, sep="")

    #print(df.iloc[-1])

    print("Writing binary C struct data to", fname_bin)
    t = time.time()
    f = open(fname_bin, "ba")
    for _, row in df.iterrows():
        f.write(struct.pack(fmt, *row))
    f.close()
    print("Wrote", len(df.index), "records in %.3f s" % (time.time() - t))

    print("Writing CSV text data to", fname_csv)
    t = time.time()
    df["pums_id"] = df["pums_id"].astype("string")
    df.to_csv(fname_csv)
    print("Wrote", len(df.index), "records in %.3f s" % (time.time() - t))


def process_census_bg_shape_file(dir_names, geoid_locs_map):
    for dname in dir_names:
        # remove trailing slash if it exists
        if dname[-1] == "/":
            dname = dname[:-1]
        shape_fname = dname + "/" + os.path.split(dname)[1] + ".shp"
        # don't actually need to compute the centroid because the block group file has it under the INTPTLAT10 and INTPTLON10 cols
        #df = geopandas.read_file(shape_fname, include_fields=["GEOID10", "geometry"])
        #df = df.to_crs(crs=4326)
        #df["centroid"] = df.centroid
        print("Reading shape files at", shape_fname)
        df = geopandas.read_file(shape_fname, include_fields=["GEOID10", "INTPTLAT10", "INTPTLON10"], ignore_geometry=True)
        df.GEOID10 = df.GEOID10.astype("int64")
        df.INTPTLAT10 = df.INTPTLAT10.astype("float32")
        df.INTPTLON10 = df.INTPTLON10.astype("float32")
        df.to_csv(shape_fname + ".csv")
        print("Wrote", len(df.index), "GEOID locations to", shape_fname + ".csv")
        #print(df.dtypes)
        geoid_locs_map.update(df.set_index("GEOID10").T.to_dict("list"))


if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser(description="Convert UrbanPop feather files to C++ struct binary file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--files", "-f", required=True, nargs="+", help="Feather files")
    parser.add_argument("--shape_files_dir", "-s", required=True, nargs="+",
                        help="Directories for census block group shape files. Available from\n" + \
                        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Block+Groups")
    args = parser.parse_args()

    geoid_locs_map = {}
    process_census_bg_shape_file(args.shape_files_dir, geoid_locs_map)
    print("GEOID to locations map contains", len(geoid_locs_map), "entries")

    first = True
    for fname in args.files:
        process_feather_file(fname, args.output, geoid_locs_map, first)
        first = False

    print("Processed", len(args.files), "files in %.2f s" % (time.time() - t))
