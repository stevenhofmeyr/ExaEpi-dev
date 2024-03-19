#include <iostream>
#include <fstream>

#include "UrbanPopAgentStruct.H"

using namespace std;

int main(int argc, char** argv) {
    ifstream f(argv[1], std::ios::binary);
    int lines = 0;
    for (;; lines++) {
        UrbanPopAgent agent;
        agent.read_binary(f);
        //cout << agent << endl;
        if (f.peek() == EOF) break;
    }
    cout << "Read " << lines << " lines\n";
    return 0;
}
