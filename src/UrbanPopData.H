/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <string>

namespace UrbanPop {

struct Person {
    int64_t geoid;    // the census block group ID - determines the community number
    int p_id;         // the unique person/agent id
    int h_id;         // the ID of the household this agent belongs to - this is the same as the family ID
    int pr_age;       // age of this agent
    int pr_emp_stat;  // whether employed or not
    int pr_commute;   // commute distance

    void set(int64_t geoid, int p_id, int h_id, int pr_age, int pr_emp_stat, int pr_commute);
};

struct BlockGroup {
    int64_t geoid;
    float latitude;
    float longitude;
    size_t file_offset;
    int population;
    int box_i;
    int num_households;
    int num_employed;
    int num_military;
    amrex::Vector<Person> people;

    bool read(std::istringstream &iss);
    bool read_people(std::ifstream &f);
};

struct UrbanPopData {
    /*! \brief Null constructor */
    UrbanPopData () {}

    void construct_geom(const std::string &fname, amrex::Geometry &geom, amrex::DistributionMapping &dm, amrex::BoxArray &ba);

    void InitFromFile (const std::string& fname, amrex::Geometry &geom, amrex::DistributionMapping &dm, amrex::BoxArray &ba);

    void print () const;

    // my_ = for this process
    int my_num_block_groups;  // each block group is a community of around 1500 people
    int my_num_agents;        // number of people
    // across all processes
    int all_num_block_groups;
    int all_num_agents;

    amrex::Vector<BlockGroup> block_groups;

    // the vectors have one entry per agent
    amrex::Vector<int64_t> geoid;    // the census block group ID - determines the community number
    amrex::Vector<int> p_id;         // the unique person/agent id
    amrex::Vector<int> h_id;         // the ID of the household this agent belongs to - this is the same as the family ID
    amrex::Vector<int> pr_age;       // age of this agent
    amrex::Vector<int> pr_emp_stat;  // whether employed or not
    amrex::Vector<int> pr_commute;   // commute distance
};

}