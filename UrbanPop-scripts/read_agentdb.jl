using SHA, Mmap

struct Agent
    pums_id::UInt64            # pums household id
    fips_code::UInt64          # fips block-group code

    household_id::UInt32       # household id (index, unique per state)
    person_id::UInt32          # person / agent id (index, unique per state)

    household_income::UInt32   # $/year 0:999_999_999

    person_commute_time::Int16 # minutes[-999, 1:200], -999 -> non-worker

    person_naics::UInt16       # 3 digit NAICS occupation code

    household_size::UInt8      # 1:20
    household_type::UInt8
    #=  0: group quarters
        1: Single-family detached dwelling
        2: Single-family attached dwelling
        3: Multi-family dwelling (2 units)
        4: Multi-family dwelling (3-4 units)
        5: Multi-family dwelling (5-9 units)
        6: Multi-family dwelling (10-19 units)
        7: Multi-family dwelling (20-49 units)
        8: Multi-family dwelling (>=50 units)
        9: Mobile Home
        10: Boat, RV, van, etc.
    =#
    householder_age::UInt8            # 0:99 years - age of
    household_kids::UInt8             # 0: no, 1: yes - are there kids in this household?
    household_workers::UInt8          # 0:20 - total # of workers in the household
    household_nonworkers::UInt8       # 0:20 - total # of non-workers in household
    household_adult_workers::UInt8    # 0:20 - # of adult workers
    household_adult_nonworkers::UInt8 # 0:20 - # of adult non-workers

    household_arrangement::UInt8
    #=
        0: NA
        1: married
        2: male_no_spouse
        3: female_no_spouse
        4: alone
        5: not_alone
    =#
    household_tenure::UInt8
    #=
        0: NA
        1: own
        2: rent
        3: other
    =#
    household_vehicles::UInt8
    #=
        [0, 1, 2, 3, 4, 5]
        6: 6+
        7: missing
    =#

    person_age::UInt8  # 0:99 years
    person_sex::UInt8  # 0: male, 1: female
    person_race::UInt8
    #=  0: race_white
        1: race_blk_af_amer
        2: race_asian
        3: race_native_amer
        4: race_pac_island
        5: race_other
        6: race_mult
    =#
    person_ethnicity::UInt8 # 0: non-hispanic/latinx, 1: hispanic/latinx
    person_commute_mode::UInt8
    #=  0: none
        1: car_truck_van
        2: public_transportation
        3: bicycle
        4: walked
        5: motorcycle
        6: taxicab
        7: other
        8: work from home
    =#

    person_commute_occupancy::UInt8
    #=
        0: NA
        1: drove alone
        2: carpooled
    =#

    person_ipr::UInt8 # income:poverty ratio
    #=
        0: < 0.5       (L050)
        1: 0.5 - 0.99  (050_099)
        2: 1.0 - 1.24  (100_124)
        3: 1.25 - 1.49 (125_149)
        4: 1.5 - 1.84  (150_184)
        5: 1.85 - 1.99 (185_199)
        6: > 2.0       (GE200)
    =#

    person_employment::UInt8
    #=
        0: ""
        1: not.in.force
        2: unemployed
        3: employed
        4: military
    =#

    person_school_grade::UInt8
    #=
        0: NA
        1: preschl
        2: kind
        3: 1st
        4: 2nd
        5: 3rd
        6: 4th
        7: 5th
        8: 6th
        9: 7th
        10: 8th
        11: 9th
        12: 10th
        13: 11th
        14: 12th
        15: undergrad
        16: grad
    =#

end

struct Tract
    # index of the first agent belonging to this tract w/in the corresponding
    # agents.bin file
    index::UInt64

    n_agent::UInt64   # \# of agents that reside in this tract
    fips_code::UInt64 # 11 digit tract level FIPS code for this tract
end

function memmap_tract(ifile::AbstractString)

    version = sha2_256(join(map(string, fieldnames(Tract))))

    open(ifile, "r") do io
        tmp = read(io, 32)
        @assert(tmp == version, "Version mismatch between current struct layout and version on disk!")
        n = Base.read(io, UInt64)
        # skip the next 8 bytes (strust size, n-tract, n-household)
        seek(io, position(io) + 8)
        return Mmap.mmap(io, Vector{Tract}, (n,), grow=false)
    end
end


function memmap_agent(ifile::AbstractString)
    version = sha2_256(join(map(string, fieldnames(Agent))))
    open(ifile, "r") do io
        tmp = read(io, 32)
        @assert(tmp == version, "Version mismatch between current struct layout and version on disk!")
        n = Base.read(io, UInt64)
        # skip the next 8 bytes (strust size, n-tract, n-household)
        seek(io, position(io) + 8)
        return Mmap.mmap(io, Vector{Agent}, (n,), grow=false)
    end
end

fname = ARGS[1]

println("Reading urbanpop data from ", fname)

agents = memmap_agent(fname)

print(size(agents), "\n")

agent = agents[1]

println(agent.pums_id)
println(agent.fips_code)

dump(agent)
