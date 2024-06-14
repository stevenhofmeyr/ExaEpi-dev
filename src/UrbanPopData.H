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
    int box_i;
    int x;
    int y;
    int num_households;
    int num_employed;
    int num_military;
    amrex::Vector<Person> people;

    bool read(std::istringstream &iss);
    bool read_people(std::ifstream &f);
};

struct UrbanPopData {
    // each block group has around 1500 people
    amrex::Vector<BlockGroup> block_groups;

    /*! \brief Null constructor */
    UrbanPopData () {}

    void construct_geom(const std::string &fname, amrex::Geometry &geom, amrex::DistributionMapping &dm, amrex::BoxArray &ba);

    void InitFromFile(const std::string& fname, amrex::Geometry &geom, amrex::DistributionMapping &dm, amrex::BoxArray &ba);

    void print() const;

};

}