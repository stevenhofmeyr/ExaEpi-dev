/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <string>

namespace UrbanPop {

struct Person {
    int64_t h_geoid;  // the census block group ID for home - determines the community number
    int64_t w_geoid;  // the census block group ID for home - determines the community number
    int work_x, work_y; // work location
    int p_id;         // the unique person/agent id
    int h_id;         // the ID of the household this agent belongs to - this is the same as the family ID
    int pr_age;       // age of this agent
    int pr_emp_stat;  // whether employed or not
    int pr_commute;   // commute distance

    void set(int64_t geoid, int64_t w_geoid, int work_x, int work_y, int p_id, int h_id, int pr_age,
             int pr_emp_stat, int pr_commute);

    bool is_worker();
};

struct BlockGroup {
    int64_t geoid;
    float lng;
    float lat;
    size_t file_offset;
    int box_i;
    int x;
    int y;
    int num_households;
    int num_employed;
    int num_military;
    int num_workers;
    amrex::Vector<Person> people;

    bool read(std::istringstream &iss);
    bool read_people(std::ifstream &f, float min_lng, float min_lat, float gspacing_x, float gspacing_y);
};

struct UrbanPopData {
    // each block group has around 1500 people
    amrex::Vector<BlockGroup> block_groups;
    std::unordered_map<int64_t, int> block_group_workers;
    float min_lng, min_lat;
    float gspacing_x, gspacing_y;

    /*! \brief Null constructor */
    UrbanPopData () {}

    void construct_geom(const std::string &fname, amrex::Geometry &geom, amrex::DistributionMapping &dm, amrex::BoxArray &ba);

    void InitFromFile(const std::string& fname, amrex::Geometry &geom, amrex::DistributionMapping &dm, amrex::BoxArray &ba);

    void print() const;

};

}