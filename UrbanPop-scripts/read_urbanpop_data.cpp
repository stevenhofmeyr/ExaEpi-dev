#include <iostream>
#include <fstream>
#include <regex>
#include <string_view>
#include <unordered_map>

using namespace std;

const vector<string> HH_TYPES = {"hh", "gq"};
const vector<string> LIVING_ARRANGEMENTS = {"married", "male_no_spouse", "female_no_spouse", "alone", "not_alone"};
const vector<string> HH_DWGS = {"single_fam_detach", "single_fam_attach", "2_unit",    "3_4_unit", "5_9_unit",
                                "10_19_unit",        "20_49_unit",        "GE50_unit", "mob_home", "other"};
const vector<string> HH_TENURES = {"own", "rent", "other"};
const vector<string> PR_SEXES = {"female", "male"};
const vector<string> PR_RACES = {"white", "blk_af_amer", "asian", "native_amer", "pac_island", "other", "mult"};
const vector<string> PR_HSPLATS = {"no", "yes"};
const vector<string> PR_IPRS = {"L050", "050_099", "100_124", "125_149", "150_184", "185_199", "GE200"};
const vector<string> PR_EMPLOYMENTS = {"not.in.force", "unemp", "employed", "mil"};
const vector<string> HH_TRAVELS = {
    "car_truck_van", "public_transportation", "bicycle", "walked", "motorcycle", "taxicab", "other", "wfh"};
const vector<string> PR_VEH_OCCS = {"drove_alone", "carpooled"};
const vector<string> PR_GRADES = {"preschl", "kind", "1st", "2nd",  "3rd",  "4th",  "5th",       "6th",
                                  "7th",     "8th",  "9th", "10th", "11th", "12th", "undergrad", "grad"};

vector<string> split_string (const string& in_pattern, const string& content) {
    vector<string> split_content;
    regex pattern(in_pattern);
    copy(sregex_token_iterator(content.begin(), content.end(), pattern, -1), sregex_token_iterator(),
         back_inserter(split_content));
    return split_content;
}

uint8_t get_option_index (const string& s, const vector<string>& options, const string& option_name) {
    if (s.empty()) return options.size();
    auto it = find(options.begin(), options.end(), s);
    if (it != options.end()) return it - options.begin();
    cerr << "Unknown " << option_name << ": " << s << endl;
    abort();
    return 0;
}

uint8_t get_option_uint8 (const string& s) {
    if (s.empty()) return 0;
    try {
        return stoi(s);
    } catch (const invalid_argument& e) {
        cerr << "invalid argument " << e.what() << " for string \"" << s << "\"\n";
        return 0;
    }
}

uint32_t get_option_uint32 (const string& s) {
    if (s.empty()) return 0;
    try {
        return stoi(s);
    } catch (const invalid_argument& e) {
        cerr << "invalid argument " << e.what() << " for string \"" << s << "\"\n";
        return 0;
    }
}

struct Agent {
    uint32_t p_id;
    uint64_t pums_id;
    uint32_t h_id;
    uint64_t geoid;
    uint8_t hh_size;
    uint8_t hh_type = 0; // 1 = hh, 2 = gq
    uint8_t hh_living_arrangement;
    uint8_t hh_age;
    bool hh_has_kids;
    uint32_t hh_income;
    uint8_t hh_nb_wrks;
    uint8_t hh_nb_non_wrks;
    uint8_t hh_nb_adult_wrks;
    uint8_t hh_nb_adult_non_wrks;
    uint8_t hh_dwg;
    uint8_t hh_tenure;
    uint8_t hh_vehicles;
    uint8_t pr_age;
    uint8_t pr_sex;
    uint8_t pr_race;
    bool pr_hsplat;
    uint8_t pr_ipr;
    string pr_naics;
    uint8_t pr_emp_stat;
    uint8_t pr_travel;
    uint8_t pr_veh_occ;
    uint8_t pr_commute;
    uint8_t pr_grade;

    Agent(const string& line) {
        const int NUM_TOKENS = 29;
        auto tokens = split_string(",", line);
        // can have one less tokens if the last column is not set
        if (tokens.size() != NUM_TOKENS && tokens.size() != NUM_TOKENS - 1) {
            cerr << "Incorrect number of tokens in line, expected " << NUM_TOKENS << " but found " << tokens.size() << endl;
            cerr << "\"" << line << "\"" << endl;
            abort();
        }
        string pums_id_str = tokens[2];
        pums_id = stoull(pums_id_str);
        geoid = stoull(tokens[4]);
        p_id = stoul(tokens[1].substr(pums_id_str.length() + 1));
        h_id = stoul(tokens[3].substr(pums_id_str.length() + 1));
        hh_size = get_option_uint8(tokens[5]);
        hh_type = get_option_index(tokens[6], HH_TYPES, "hh_type");
        hh_living_arrangement = get_option_index(tokens[7], LIVING_ARRANGEMENTS, "hh_living_arrangement");
        hh_age = get_option_uint8(tokens[8]);
        hh_has_kids = (tokens[9] == "no" ? false : true);
        hh_income = get_option_uint32(tokens[10]);
        hh_nb_wrks = get_option_uint8(tokens[11]);
        hh_nb_non_wrks = get_option_uint8(tokens[12]);
        hh_nb_adult_wrks = get_option_uint8(tokens[13]);
        hh_nb_adult_non_wrks = get_option_uint8(tokens[14]);
        hh_dwg = get_option_index(tokens[15], HH_DWGS, "hh_dwg");
        hh_tenure = get_option_index(tokens[16], HH_TENURES, "hh_tenure");
        hh_vehicles = (tokens[17] == "GE06" ? 6 : get_option_uint8(tokens[17]));
        pr_age = get_option_uint8(tokens[18]);
        pr_sex = get_option_index(tokens[19], PR_SEXES, "pr_sex");
        pr_race = get_option_index(tokens[20], PR_RACES, "pr_race");
        pr_hsplat = get_option_index(tokens[21], PR_HSPLATS, "pr_hsplat");
        pr_ipr = get_option_index(tokens[22], PR_IPRS, "pr_ipr");
        pr_naics = tokens[23];
        pr_emp_stat = get_option_index(tokens[24], PR_EMPLOYMENTS, "pr_employment");
        pr_travel = get_option_index(tokens[25], HH_TRAVELS, "hh_travel");
        pr_veh_occ = get_option_index(tokens[26], PR_VEH_OCCS, "pr_veh_occ");
        pr_commute = get_option_uint8(tokens[27]);
        pr_grade = (tokens.size() == NUM_TOKENS ? get_option_index(tokens[28], PR_GRADES, "pr_grade") : PR_GRADES.size());
    }

    friend ostream& operator<<(ostream& os, const Agent& agent) {
        os << agent.p_id << "\t" << agent.pums_id << "\t" << agent.h_id << "\t" << agent.geoid << "\t" << (int) agent.hh_size
           << "\t" << (int) agent.hh_type << "\t" << (int) agent.hh_living_arrangement << "\t" << (int) agent.hh_age << "\t"
           << agent.hh_has_kids << "\t" << agent.hh_income << "\t" << (int) agent.hh_nb_wrks << "\t" << (int) agent.hh_nb_non_wrks
           << "\t" << (int) agent.hh_nb_adult_wrks << "\t" << (int) agent.hh_nb_adult_non_wrks << "\t" << (int) agent.hh_dwg
           << "\t" << (int) agent.hh_tenure << "\t" << (int) agent.hh_vehicles << "\t" << (int) agent.pr_age << "\t"
           << (int) agent.pr_sex << "\t" << (int) agent.pr_race << "\t" << (int) agent.pr_hsplat << "\t" << agent.pr_ipr << "\t"
           << agent.pr_naics << "\t" << (int) agent.pr_emp_stat << "\t" << (int) agent.pr_travel << "\t" << (int) agent.pr_veh_occ
           << "\t" << (int) agent.pr_commute << "\t" << (int) agent.pr_grade;
        return os;
    }
};

vector<Agent> read_csv (const string& fname) {
    const string HEADER =
        ",p_id,pums_id,h_id,geoid,hh_size,hh_type,hh_living_arrangement,hh_age,hh_has_kids,hh_income,hh_nb_wrks,hh_nb_non_wrks,"
        "hh_nb_adult_wrks,hh_nb_adult_non_wrks,hh_dwg,hh_tenure,hh_vehicles,pr_age,pr_sex,pr_race,pr_hsplat,pr_ipr,pr_naics,"
        "pr_emp_stat,pr_travel,pr_veh_occ,pr_commute,pr_grade";
    ifstream f(fname);
    if (!f.is_open()) {
        cerr << "Cannot open file " << fname << endl;
        abort();
    }
    string line;
    vector<Agent> agents;
    unordered_map<uint64_t, int> geoid_counts;
    int line_num = 0;
    while (getline(f, line)) {
        // first line is headers. check that they are as expected so there is no version mismatch
        if (line_num == 0) {
            if (line != HEADER) {
                cerr << "Mismatch in file header:\nExpected " << HEADER << "\nRead    " <<  line << endl;
                abort();
            }
        } else {
            agents.emplace_back(Agent(line));
            auto geoid = agents.back().geoid;
            auto it = geoid_counts.find(geoid);
            if (it == geoid_counts.end()) geoid_counts.insert({geoid, 1});
            else it->second++;
        }
        line_num++;
    }
    cout << "Found " << geoid_counts.size() << " unique geoids\n";
    ofstream ofs("geoid-counts.txt");
    for (auto &[geoid, count]: geoid_counts) {
        ofs << geoid << " " << count << endl;
    }
    ofs.close();
    return agents;
}

void write_binary(const string &fname, const vector<Agent> &agents) {
    ofstream ofile(fname, ios::binary);
    auto num_agents = agents.size();
    ofile.write(reinterpret_cast<const char*>(&num_agents), sizeof(num_agents));
    for (const auto& agent : agents) {
        ofile.write(reinterpret_cast<const char*>(&agent), sizeof(agent));
    }
}

int main (int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: read_urbanpop_data <filename>\n";
        return 0;
    }
    auto agents = read_csv(argv[1]);
    write_binary(string(argv[1]) + ".bin", agents);
    return 0;
}
