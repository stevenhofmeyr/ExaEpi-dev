#include <iostream>
#include <fstream>
#include <vector>

#include "UrbanPopAgentStruct.H"

using namespace std;

int main(int argc, char** argv) {
    ifstream f(argv[1]);
    if (!f) {
        cerr << "Error: could not open " << argv[1] << endl;
        return 1;
    }
    string buf;
    // first line is the headings
    if (!getline(f, buf)) {
        cerr << "Error: could not read " << argv[1] << endl;
        return 1;
    }
    int line = 0;
    vector<UrbanPopAgent> agents;
    for (;; line++) {
        UrbanPopAgent agent;
        try {
            if (!agent.read_csv(f)) break;
        } catch (const std::exception &ex) {
            cerr << "Error on line " << line << ": " << ex.what() << endl;
            throw ex;
        }
        cout << "[" << line << "] " << agent << endl;
        agents.push_back(agent);
        if (line == 10) break;
    }
    //cout << agents.back() << endl;
    cout << "Read " << line << " lines\n";
    return 0;
}
