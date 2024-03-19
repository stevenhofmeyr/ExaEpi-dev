#!/usr/bin/env python

import sys
import os.path
import struct
import pandas
from pandas.api.types import CategoricalDtype
import numpy as np
import time

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

fname = sys.argv[1]
print("Reading data from", fname)
t = time.time()
df = pandas.read_feather(fname)
print("Read", len(df.index), "records in %.3f s" % (time.time() - t))

# all the fields parsed, in order
df.p_id = df.p_id.str.split("-").str[-1].astype("int32")
PUMS_ID_LEN = 13
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
NAICS_LEN = 8
df.pr_naics = df.pr_naics.map(lambda x: str(x).ljust(NAICS_LEN)).str.encode("utf-8")#.astype("string")
max_naics_len = df.pr_naics.map(len).max()
if max_naics_len != NAICS_LEN:
    print("Incorrect max PR_NAICS length", max_naics_len, "!=", NAICS_LEN, file=sys.stderr)
df.pr_emp_stat = df.pr_emp_stat.astype(pr_emp_stat).cat.codes
df.pr_travel = df.pr_travel.astype(pr_travel).cat.codes
df.pr_veh_occ = df.pr_veh_occ.astype(pr_veh_occ).cat.codes
df.pr_commute = df.pr_commute.astype("int8")
df.pr_grade = df.pr_grade.astype(pr_grade).cat.codes

fname, extension = os.path.splitext(fname)
fname_bin = fname + ".bin"
fname_csv = fname + ".csv"

#print("Unique PUMS", len(df.pums_id.unique()), "max length", df.pums_id.map(len).max())
#print("Unique NAICS", len(df.pr_naics.unique()), "max length", df.pr_naics.map(len).max())

correct_dtypes = df.dtypes
# first record write -99 for all integer fields, and dashes for strings
check_record = pandas.DataFrame([[-99, b'-' * PUMS_ID_LEN, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99
                                  -99, -99, -99, -99, -99, -99, -99, b'-' * NAICS_LEN, -99, -99, -99, -99, -99]], columns=df.columns)
# coerce to correct types
check_record = check_record.astype(correct_dtypes)
df = pandas.concat([check_record, df], ignore_index=True)

# pack structure by moving int32_t value to before all int8_t values
df.insert(4, "hh_income", df.pop("hh_income"))

# move char arrays to end of struct
df.insert(len(df.columns) - 1, "pums_id", df.pop("pums_id"))
df.insert(len(df.columns) - 1, "pr_naics", df.pop("pr_naics"))

#print(df.dtypes)

# print out header file for C++ import
f_hdr = open("UrbanPopAgentStruct.H", "w")
print("#include <fstream>\n", file=f_hdr)
print("struct UrbanPopAgent {", file=f_hdr)

fmt = ""
read_func_str = ""
write_str = ""

for i in range(len(df.columns)):
    if df.columns[i] == 'pums_id':
        fmt += str(PUMS_ID_LEN) + 's'
        print("    char " + df.columns[i] + "[" + str(PUMS_ID_LEN) + "];", file=f_hdr)
        write_str += "        os << \"b'\" << std::string(agent." + df.columns[i] + ", " + str(PUMS_ID_LEN) + ") << \"',\";\n"
    elif df.columns[i] == 'pr_naics':
        fmt += str(NAICS_LEN) + 's'
        print("    char " + df.columns[i] + "[" + str(NAICS_LEN) + "];", file=f_hdr)
        write_str += "        os << \"b\'\" << std::string(agent." + df.columns[i] + ", " + str(NAICS_LEN) + ") << \"'\";\n"
    elif df.dtypes[i] == np.int8:
        fmt += 'b'
        print("    int8_t " + df.columns[i] + ";", file=f_hdr)
    elif df.dtypes[i] == np.int32:
        fmt += 'i'
        print("    int32_t " + df.columns[i] + ";", file=f_hdr)
    elif df.dtypes[i] == np.int64:
        fmt += 'q'
        print("    int64_t " + df.columns[i] + ";", file=f_hdr)
    else:
        raise RuntimeError("ERRROR: dtype", df.dtypes[i], "in column", df.columns[i], "doesn't match a format string")
    if df.columns[i] in ["pums_id", "pr_naics"]:
        read_func_str += "        f.read(" + df.columns[i] + ", sizeof(" + df.columns[i] + "));\n"
    else:
        read_func_str += "        f.read((char*)&" + df.columns[i] + ", sizeof(" + df.columns[i] + "));\n"
        write_str += "        os << (int64_t)agent." + df.columns[i] + " << ',';\n"

print("\n    void read_binary(std::ifstream &f) {", file=f_hdr)
print(read_func_str, file=f_hdr, end='')
print("    }\n", file=f_hdr)

print("\n    friend std::ostream& operator<<(std::ostream& os, const UrbanPopAgent& agent) {", file=f_hdr)
print(write_str, file=f_hdr, end="")
print("        return os;", file=f_hdr)
print("    }", file=f_hdr)

print("};", file=f_hdr)

#print(df.iloc[0])

print("Writing binary C struct data to", fname_bin)
t = time.time()
f = open(fname_bin, "wb")
for _, row in df.iterrows():
    f.write(struct.pack(fmt, *row))
f.close()
print("Wrote", len(df.index), "records in %.3f s" % (time.time() - t))

print("Writing CSV text data to", fname_csv)
t = time.time()
df["pums_id"] = df["pums_id"].astype("string")
df.to_csv(fname_csv)
print("Wrote", len(df.index), "records in %.3f s" % (time.time() - t))
