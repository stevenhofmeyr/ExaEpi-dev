/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <string>

struct UrbanPopData
{
   /*! \brief Null constructor */
    UrbanPopData () {}

    void InitFromFile (const std::string& fname);

    void Print () const;

    template<typename T>
    void CopyToDeviceAsync (const amrex::Vector<T>& h_vec, amrex::Gpu::DeviceVector<T>& d_vec);

    template<typename T>
    void CopyToHostAsync (const amrex::Gpu::DeviceVector<T>& d_vec, amrex::Vector<T>& h_vec);

    void CopyDataToDevice ();

    int num_block_groups;  // each block group is a community of around 1500 people
    int num_agents;        // number of agents

    // the vectors have one entry per agent
    amrex::Vector<int64_t> geoid;    // the census block group ID - determines the community number
    amrex::Vector<int> h_id;         // the ID of the household this agent belongs to - this is the same as the family ID
    amrex::Vector<int> pr_age;       // age of this agent
    amrex::Vector<int> pr_emp_stat;  // whether employed or not
    amrex::Vector<int> pr_commute;   // commute distance

    // device copies of the above host vectors
    amrex::Gpu::DeviceVector<int64_t> geoid_d;
    amrex::Gpu::DeviceVector<int> h_id_d;
    amrex::Gpu::DeviceVector<int> pr_age_d;
    amrex::Gpu::DeviceVector<int> pr_emp_stat_d;
    amrex::Gpu::DeviceVector<int> pr_commute_d;

};