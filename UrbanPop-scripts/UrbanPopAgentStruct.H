#include <fstream>

struct UrbanPopAgent {
    int32_t p_id;
    int32_t h_id;
    int64_t geoid;
    int32_t hh_income;
    int8_t hh_size;
    int8_t hh_type;
    int8_t hh_living_arrangement;
    int8_t hh_age;
    int8_t hh_has_kids;
    int8_t hh_nb_wrks;
    int8_t hh_nb_non_wrks;
    int8_t hh_nb_adult_wrks;
    int8_t hh_nb_adult_non_wrks;
    int8_t hh_dwg;
    int8_t hh_tenure;
    int8_t hh_vehicles;
    int8_t pr_age;
    int8_t pr_sex;
    int8_t pr_race;
    int8_t pr_hsplat;
    int8_t pr_ipr;
    int8_t pr_emp_stat;
    int8_t pr_travel;
    int8_t pr_veh_occ;
    int8_t pr_commute;
    int8_t pr_grade;
    char pums_id[13];
    char pr_naics[8];

    void read_binary(std::ifstream &f) {
        f.read((char*)&p_id, sizeof(p_id));
        f.read((char*)&h_id, sizeof(h_id));
        f.read((char*)&geoid, sizeof(geoid));
        f.read((char*)&hh_income, sizeof(hh_income));
        f.read((char*)&hh_size, sizeof(hh_size));
        f.read((char*)&hh_type, sizeof(hh_type));
        f.read((char*)&hh_living_arrangement, sizeof(hh_living_arrangement));
        f.read((char*)&hh_age, sizeof(hh_age));
        f.read((char*)&hh_has_kids, sizeof(hh_has_kids));
        f.read((char*)&hh_nb_wrks, sizeof(hh_nb_wrks));
        f.read((char*)&hh_nb_non_wrks, sizeof(hh_nb_non_wrks));
        f.read((char*)&hh_nb_adult_wrks, sizeof(hh_nb_adult_wrks));
        f.read((char*)&hh_nb_adult_non_wrks, sizeof(hh_nb_adult_non_wrks));
        f.read((char*)&hh_dwg, sizeof(hh_dwg));
        f.read((char*)&hh_tenure, sizeof(hh_tenure));
        f.read((char*)&hh_vehicles, sizeof(hh_vehicles));
        f.read((char*)&pr_age, sizeof(pr_age));
        f.read((char*)&pr_sex, sizeof(pr_sex));
        f.read((char*)&pr_race, sizeof(pr_race));
        f.read((char*)&pr_hsplat, sizeof(pr_hsplat));
        f.read((char*)&pr_ipr, sizeof(pr_ipr));
        f.read((char*)&pr_emp_stat, sizeof(pr_emp_stat));
        f.read((char*)&pr_travel, sizeof(pr_travel));
        f.read((char*)&pr_veh_occ, sizeof(pr_veh_occ));
        f.read((char*)&pr_commute, sizeof(pr_commute));
        f.read((char*)&pr_grade, sizeof(pr_grade));
        f.read(pums_id, sizeof(pums_id));
        f.read(pr_naics, sizeof(pr_naics));
    }


    friend std::ostream& operator<<(std::ostream& os, const UrbanPopAgent& agent) {
        os << (int64_t)agent.p_id << ',';
        os << (int64_t)agent.h_id << ',';
        os << (int64_t)agent.geoid << ',';
        os << (int64_t)agent.hh_income << ',';
        os << (int64_t)agent.hh_size << ',';
        os << (int64_t)agent.hh_type << ',';
        os << (int64_t)agent.hh_living_arrangement << ',';
        os << (int64_t)agent.hh_age << ',';
        os << (int64_t)agent.hh_has_kids << ',';
        os << (int64_t)agent.hh_nb_wrks << ',';
        os << (int64_t)agent.hh_nb_non_wrks << ',';
        os << (int64_t)agent.hh_nb_adult_wrks << ',';
        os << (int64_t)agent.hh_nb_adult_non_wrks << ',';
        os << (int64_t)agent.hh_dwg << ',';
        os << (int64_t)agent.hh_tenure << ',';
        os << (int64_t)agent.hh_vehicles << ',';
        os << (int64_t)agent.pr_age << ',';
        os << (int64_t)agent.pr_sex << ',';
        os << (int64_t)agent.pr_race << ',';
        os << (int64_t)agent.pr_hsplat << ',';
        os << (int64_t)agent.pr_ipr << ',';
        os << (int64_t)agent.pr_emp_stat << ',';
        os << (int64_t)agent.pr_travel << ',';
        os << (int64_t)agent.pr_veh_occ << ',';
        os << (int64_t)agent.pr_commute << ',';
        os << (int64_t)agent.pr_grade << ',';
        os << "b'" << std::string(agent.pums_id, 13) << "',";
        os << "b'" << std::string(agent.pr_naics, 8) << "'";
        return os;
    }
};
